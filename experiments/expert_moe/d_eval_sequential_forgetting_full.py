import sys
import gc
import json
import argparse
import torch
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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

TRAIN_ORDER = ["en", "de", "nl", "fr"]
MODEL_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "sequential_forgetting_full"
RESULTS_ROOT = ROOT / "outputs" / "expert_moe" / "results" / "sequential_forgetting_full"
GLOSSARY_PATH = ROOT / "data" / "term" / "npd_glossary_multi.json"


def load_test_data(lang):
    with open(ROOT / CFG["data"][lang]["test"]) as f:
        return json.load(f)


def run_inference(model, tokenizer, data, src_lang, max_samples):
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


def evaluate_seed(seed, max_samples, use_comet):
    final_model_path = MODEL_ROOT / f"seed{seed}" / f"step4_{TRAIN_ORDER[-1]}" / "final_model"
    if not final_model_path.exists():
        print(f"  Skipping seed={seed}: model not found at {final_model_path}")
        return

    print(f"  Loading step4 final model  seed={seed}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Full FT: load entire model directly, no PEFT wrapper
    model = AutoModelForSeq2SeqLM.from_pretrained(
        str(final_model_path),
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(str(final_model_path))

    output_dir = RESULTS_ROOT / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results, pred_dump = {}, {}

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

        evaluator = FTAEvaluator(str(GLOSSARY_PATH), src_lang=lang, tgt_lang="no", use_comet=use_comet)
        metrics = evaluator.evaluate_all(sources, preds, refs)
        results[lang] = metrics
        pred_dump[lang] = {"inputs": sources, "predictions": preds, "references": refs}

        print(
            f"  seq_forget_full {lang} seed={seed}: "
            f"bleu={metrics.get('bleu', 0)*100:.1f}  "
            f"chrf={metrics.get('chrf', 0):.1f}  "
            f"comet={metrics.get('comet', 0):.3f}  "
            f"fta_sent={metrics.get('fta_mean_sentence', 0):.3f}"
        )

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(pred_dump, f, ensure_ascii=False, indent=2)

    train_summary = MODEL_ROOT / f"seed{seed}" / "forgetting_summary.json"
    if train_summary.exists():
        with open(train_summary) as f:
            summary = json.load(f)
        with open(output_dir / "forgetting_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  delta_forget_bleu={summary.get('delta_forget_bleu', 0)*100:.2f}")

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  Saved: {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    parser.add_argument("--max_samples", type=int, default=999999)
    parser.add_argument("--use_comet", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        print(f"\nEvaluating sequential forgetting full FT (step4 final model)  seed={seed}")
        evaluate_seed(seed, args.max_samples, args.use_comet)

    print("\nDone.")


if __name__ == "__main__":
    main()