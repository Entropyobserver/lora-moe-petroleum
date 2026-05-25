import sys
import json
import argparse
import torch
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
from evaluation.fta_evaluator import FTAEvaluator

CONFIG_PATH = Path(__file__).parent / "expert_moe.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

BACKBONE = CFG["backbone"]
LANG_CODES = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "nl": "nld_Latn",
    "no": "nob_Latn",
}

MODEL_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "multitask_lora"
RESULTS_ROOT = ROOT / "outputs" / "expert_moe" / "results" / "multitask_lora"
GLOSSARY_PATH = ROOT / "data" / "term" / "npd_glossary_multi.json"


def load_test_data(lang: str) -> list:
    with open(ROOT / CFG["data"][lang]["test"]) as f:
        return json.load(f)


def run_inference(model, tokenizer, data: list, src_lang: str, max_samples: int) -> list:
    device = next(model.parameters()).device
    tokenizer.src_lang = LANG_CODES[src_lang]
    forced_bos = tokenizer.convert_tokens_to_ids(LANG_CODES["no"])
    preds = []
    for sample in data[:max_samples]:
        inputs = tokenizer(sample["source"], return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=forced_bos, num_beams=5, max_length=128)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
    return preds


def evaluate_for_seed(seed: int, max_samples: int, use_comet: bool):
    model_path = MODEL_ROOT / f"seed{seed}" / "final_model"
    if not model_path.exists():
        print(f"  Model not found: {model_path}")
        return

    print(f"  Loading multitask LoRA  seed={seed}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BACKBONE,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, str(model_path)).to(device)
    model.eval()

    output_dir = RESULTS_ROOT / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    pred_dump = {}

    for lang in ["en", "de", "fr", "nl"]:
        test_path = ROOT / CFG["data"][lang]["test"]
        if not test_path.exists():
            print(f"  Skipping {lang}: test data not found")
            continue

        data = load_test_data(lang)
        preds = run_inference(model, tokenizer, data, lang, max_samples)
        samples = data[:max_samples]
        sources = [x["source"] for x in samples]
        refs = [x["target"] for x in samples]

        fta = FTAEvaluator(str(GLOSSARY_PATH), src_lang=lang, tgt_lang="no", use_comet=use_comet)
        metrics = fta.evaluate_all(sources, preds, refs)

        results[lang] = metrics
        pred_dump[lang] = {"inputs": sources, "predictions": preds, "references": refs}
        print(
            f"  multitask {lang} seed={seed}: "
            f"bleu={metrics.get('bleu', 0)*100:.1f}  "
            f"chrf={metrics.get('chrf', 0):.1f}  "
            f"comet={metrics.get('comet', 0):.3f}  "
            f"fta_sent={metrics.get('fta_mean_sentence', 0):.3f}"
        )

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(pred_dump, f, ensure_ascii=False, indent=2)
    print(f"  Saved: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    parser.add_argument("--max_samples", type=int, default=999999)
    parser.add_argument("--use_comet", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        print(f"\nEvaluating multitask LoRA  seed={seed}")
        evaluate_for_seed(seed, args.max_samples, args.use_comet)


if __name__ == "__main__":
    main()