import sys
import gc
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
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
ROUTER_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "langid_router"
RESULTS_ROOT = ROOT / "outputs" / "expert_moe" / "results" / "langid_router"
GLOSSARY_PATH = ROOT / "data" / "term" / "npd_glossary_multi.json"


class LangAwareRouter(nn.Module):

    def __init__(self, encoder_dim=1024, hidden_dim=256, num_experts=4):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(encoder_dim + num_experts, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, encoder_hidden, lang_onehot):
        pooled = encoder_hidden.mean(dim=1).to(next(self.network.parameters()).dtype)
        logits = self.network(torch.cat([pooled, lang_onehot.to(pooled)], dim=-1))
        return F.softmax(logits, dim=-1), logits


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


def run_inference(moe, router, data, src_lang, max_samples, device):
    tokenizer = moe.tokenizer
    tokenizer.src_lang = LANG_CODES[src_lang]
    lang_onehot = F.one_hot(
        torch.tensor([LANGS.index(src_lang)]), num_classes=len(LANGS)
    ).float().to(device)

    preds, routing_log = [], []

    for sample in data[:max_samples]:
        inputs = tokenizer(sample["source"], return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            enc_out = moe.backbone.get_encoder()(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                return_dict=True,
            )
            weights, _ = router(enc_out.last_hidden_state, lang_onehot)
            top_expert_lang = moe.expert_langs[weights.mean(dim=0).argmax().item()]
            routing_log.append({
                "weights": weights.squeeze(0).tolist(),
                "top_expert": top_expert_lang,
            })
            moe.peft_model.set_adapter(top_expert_lang)
            out = moe.peft_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_CODES["no"]),
                num_beams=5,
                max_length=128,
            )
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))

    return preds, routing_log


def evaluate_seed(seed, max_samples, use_comet):
    output_dir = RESULTS_ROOT / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    router_path = ROUTER_ROOT / f"seed{seed}" / "best_router.pt"
    if not router_path.exists():
        raise FileNotFoundError(f"Router not found: {router_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # expert_seed = seed: load same-seed experts
    moe = MixtureOfExpertsModel(
        backbone_name=BACKBONE,
        expert_model_paths=get_expert_paths(seed),
        router_hidden_dim=256,
        entropy_weight=0.01,
        device=device,
    )
    moe.eval()

    router = LangAwareRouter(encoder_dim=moe.backbone.config.hidden_size).to(device)
    router.load_state_dict(torch.load(str(router_path), map_location=device))
    router.eval()

    # Phase 1: inference on GPU
    all_data, routing_dump = {}, {}
    for lang in LANGS:
        data = load_test_data(lang)
        preds, routing_log = run_inference(moe, router, data, lang, max_samples, device)
        samples = data[:max_samples]
        all_data[lang] = {
            "sources": [x["source"] for x in samples],
            "preds": preds,
            "refs": [x["target"] for x in samples],
        }
        routing_dump[lang] = routing_log
        print(f"  langid_router {lang} seed={seed}: inference done ({len(preds)} samples)", flush=True)

    # Phase 2: free GPU before COMET
    del moe, router
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Model released from GPU, running metrics...", flush=True)

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
            f"  langid_router {lang} seed={seed}: "
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
        print(f"\nEvaluating LangID-augmented router  seed={seed}  expert_seed={seed}", flush=True)
        evaluate_seed(seed, args.max_samples, args.use_comet)

    print("\nDone.")


if __name__ == "__main__":
    main()