import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from evaluation.fta_evaluator import FTAEvaluator


CONFIG_PATH = Path(__file__).parent / "expert_moe.yaml"
with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


BACKBONE = CFG["backbone"]
LANGS = ["en", "de", "fr", "nl"]
LANG_CODES = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "nl": "nld_Latn",
    "no": "nob_Latn",
}

MODEL_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "independent_experts"
RESULTS_ROOT = ROOT / "outputs" / "expert_moe" / "results" / "independent_experts"
GLOSSARY_PATH = ROOT / "data" / "term" / "npd_glossary_multi.json"


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_test_data(lang: str) -> list:
    with open(ROOT / CFG["data"][lang]["test"], encoding="utf-8") as f:
        return json.load(f)


def run_inference(
    model_path: str,
    data: list,
    src_lang: str,
    max_samples: int,
) -> list:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BACKBONE,
        dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)

    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    tokenizer.src_lang = LANG_CODES[src_lang]
    forced_bos = tokenizer.convert_tokens_to_ids(LANG_CODES["no"])

    predictions = []
    for sample in data[:max_samples]:
        inputs = tokenizer(
            sample["source"],
            return_tensors="pt",
            truncation=True,
            max_length=128,
        ).to(device)

        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                num_beams=5,
                max_length=128,
            )

        predictions.append(tokenizer.decode(generated[0], skip_special_tokens=True))

    del model, base_model
    clear_memory()

    return predictions


def evaluate_for_seed(
    seed: int,
    max_samples: int,
    use_comet: bool,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    prediction_dump = {}

    for lang in LANGS:
        model_path = MODEL_ROOT / f"{lang}_seed{seed}" / "final_model"
        if not model_path.exists():
            print(f"Skipping {lang}: model not found")
            continue

        test_path = ROOT / CFG["data"][lang]["test"]
        if not test_path.exists():
            print(f"Skipping {lang}: test data not found")
            continue

        data = load_test_data(lang)
        predictions = run_inference(str(model_path), data, lang, max_samples)

        samples = data[:max_samples]
        sources = [sample["source"] for sample in samples]
        references = [sample["target"] for sample in samples]

        evaluator = FTAEvaluator(
            str(GLOSSARY_PATH),
            src_lang=lang,
            tgt_lang="no",
            use_comet=use_comet,
        )
        metrics = evaluator.evaluate_all(sources, predictions, references)

        results[lang] = metrics
        prediction_dump[lang] = {
            "inputs": sources,
            "predictions": predictions,
            "references": references,
        }

        print(
            f"indep_expert {lang} seed={seed}: "
            f"bleu={metrics.get('bleu', 0) * 100:.1f}  "
            f"chrf={metrics.get('chrf', 0):.1f}  "
            f"comet={metrics.get('comet', 0):.3f}  "
            f"fta_sent={metrics.get('fta_mean_sentence', 0):.3f}"
        )

    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with open(output_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(prediction_dump, f, indent=2, ensure_ascii=False)

    print(f"Saved: {output_dir}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    parser.add_argument("--max_samples", type=int, default=999999)
    parser.add_argument("--use_comet", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        print(f"\nEvaluating independent experts  seed={seed}")
        evaluate_for_seed(
            seed,
            args.max_samples,
            args.use_comet,
            RESULTS_ROOT / f"seed{seed}",
        )


if __name__ == "__main__":
    main()