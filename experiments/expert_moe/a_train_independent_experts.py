import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from model.independent_expert import ExpertTrainer


CONFIG_PATH = Path(__file__).parent / "expert_moe.yaml"
MODEL_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "independent_experts"


with open(CONFIG_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_expert(lang: str, seed: int) -> dict:
    lang_cfg = CFG["data"][lang]
    training_cfg = CFG["training"]
    lora_cfg = CFG["lora"]

    output_dir = MODEL_ROOT / f"{lang}_seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    trainer = ExpertTrainer(
        backbone_name=CFG["backbone"],
        src_lang=lang_cfg["src_lang"],
        tgt_lang="no",
    )

    result = trainer.train(
        train_path=str(ROOT / lang_cfg["train"]),
        val_path=str(ROOT / lang_cfg["val"]),
        output_dir=str(output_dir),
        lr=training_cfg["lr"],
        batch_size=training_cfg["batch_size"],
        gradient_accumulation_steps=training_cfg["gradient_accumulation"],
        num_epochs=training_cfg["num_epochs"],
        warmup_steps=training_cfg["warmup_steps"],
        eval_steps=training_cfg["eval_steps"],
        early_stopping_patience=training_cfg["early_stopping_patience"],
        weight_decay=training_cfg["weight_decay"],
        r=lora_cfg["r"],
        alpha=lora_cfg["alpha"],
        dropout=lora_cfg["dropout"],
        seed=seed,
        run_name=f"expert_{lang}_seed{seed}",
    )

    record = {
        "lang": lang,
        "seed": seed,
        "model_path": result["model_path"],
        "metrics": result["metrics"],
    }

    with open(output_dir / "training_results.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    metrics = result["metrics"]
    print(
        f"  {lang} seed={seed}: "
        f"bleu={metrics.get('bleu', 0):.4f}  "
        f"chrf={metrics.get('chrf', 0):.4f}"
    )
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train independent language-specific LoRA experts for the MoE experiment."
    )
    parser.add_argument("--langs", nargs="+", default=["en", "de", "fr", "nl"])
    parser.add_argument("--lang", type=str, default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    langs = [args.lang] if args.lang else args.langs

    all_results = []
    for lang in langs:
        train_path = ROOT / CFG["data"][lang]["train"]
        if not train_path.exists():
            print(f"Skipping {lang}: training data not found at {train_path}")
            continue

        for seed in args.seeds:
            print(f"\nTraining {lang}-NO  seed={seed}")
            all_results.append(train_expert(lang, seed))

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    summary_path = MODEL_ROOT / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()