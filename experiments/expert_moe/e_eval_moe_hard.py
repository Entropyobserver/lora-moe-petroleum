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

from model.mixture_of_experts_model import MixtureOfExpertsModel
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

LANGS = ["en", "de", "fr", "nl"]
EXPERT_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "independent_experts"
ROUTER_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "moe_router"
RESULTS_ROOT = ROOT / "outputs" / "expert_moe" / "results" / "moe_hard"
GLOSSARY_PATH = ROOT / "data" / "term" / "npd_glossary_multi.json"


def get_expert_paths(seed):
    paths = {}
    for lang in LANGS:
        p = EXPERT_ROOT / f"{lang}_seed{seed}" / "final_model"
        if not p.exists():
            raise FileNotFoundError(f"Expert not found: {p}")
        paths[lang] = str(p)
    return paths


def load_test_data(lang):
    with open(ROOT / CFG["data"][lang]["test"]) as f:
        return json.load(f)


def run_inference(moe, data, src_lang, max_samples):
    device = next(moe.parameters()).device
    tokenizer = moe.tokenizer
    tokenizer.src_lang = LANG_CODES[src_lang]
    preds, routing_log = [], []

    for sample in data[:max_samples]:
        inputs = tokenizer(sample["source"], return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            enc_out = moe.backbone.get_encoder()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            weights, _ = moe.router(enc_out.last_hidden_state.float())
            routing_log.append({
                "weights": weights.squeeze(0).tolist(),
                "top_expert": moe.expert_langs[weights.mean(dim=0).argmax().item()],
            })
            out = moe.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                tgt_lang_code=LANG_CODES["no"],
                inference_mode="hard",
                num_beams=5,
                max_length=128,
            )
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))

    return preds, routing_log


def evaluate_seed(seed, max_samples, use_comet):
    output_dir = RESULTS_ROOT / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    expert_paths = get_expert_paths(seed)
    router_path = ROUTER_ROOT / f"seed{seed}" / "best_router" / "router.pt"
    if not router_path.exists():
        raise FileNotFoundError(f"Router not found: {router_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    moe = MixtureOfExpertsModel(
        backbone_name=BACKBONE,
        expert_model_paths=expert_paths,
        router_hidden_dim=256,
        entropy_weight=0.01,
        device=device,
    )
    moe.load_router(str(router_path))
    moe.router.to(device)
    moe.eval()

    # Phase 1: inference on GPU
    all_data, routing_dump = {}, {}
    for lang in LANGS:
        data = load_test_data(lang)
        preds, routing_log = run_inference(moe, data, lang, max_samples)
        samples = data[:max_samples]
        all_data[lang] = {
            "sources": [x["source"] for x in samples],
            "preds": preds,
            "refs": [x["target"] for x in samples],
        }
        routing_dump[lang] = routing_log
        print(f"  moe_hard {lang} seed={seed}: inference done ({len(preds)} samples)", flush=True)

    # Phase 2: free GPU before COMET
    del moe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  MoE released from GPU, running metrics...", flush=True)

    # Phase 3: metrics
    results, pred_dump = {}, {}
    for lang in LANGS:
        sources = all_data[lang]["sources"]
        preds = all_data[lang]["preds"]
        refs = all_data[lang]["refs"]

        evaluator = FTAEvaluator(str(GLOSSARY_PATH), src_lang=lang, tgt_lang="no", use_comet=use_comet)
        metrics = evaluator.evaluate_all(sources, preds, refs)
        results[lang] = metrics
        pred_dump[lang] = {"inputs": sources, "predictions": preds, "references": refs}

        print(
            f"  moe_hard {lang} seed={seed}: "
            f"bleu={metrics.get('bleu', 0)*100:.1f}  "
            f"chrf={metrics.get('chrf', 0):.1f}  "
            f"comet={metrics.get('comet', 0):.3f}  "
            f"fta_sent={metrics.get('fta_mean_sentence', 0):.3f}",
            flush=True,
        )

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(output_dir / "predictions.json", "w") as f:
        json.dump(pred_dump, f, ensure_ascii=False, indent=2)
    with open(output_dir / "routing.json", "w") as f:
        json.dump(routing_dump, f, indent=2)
    print(f"  Saved: {output_dir}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    parser.add_argument("--max_samples", type=int, default=999999)
    parser.add_argument("--use_comet", action="store_true")
    args = parser.parse_args()

    for seed in args.seeds:
        print(f"\nEvaluating MoE hard routing  seed={seed}", flush=True)
        evaluate_seed(seed, args.max_samples, args.use_comet)

    print("\nDone.")


if __name__ == "__main__":
    main()