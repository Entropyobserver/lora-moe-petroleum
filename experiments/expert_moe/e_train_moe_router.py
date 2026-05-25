import sys
import json
import argparse
import os
import random
import gc
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from model.mixture_of_experts_model import MixtureOfExpertsModel
from model.router_trainer import RouterTrainer

import yaml
CONFIG_PATH = Path(__file__).parent / "expert_moe.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

BACKBONE = CFG["backbone"]
LANGS = ["en", "de", "fr", "nl"]
EXPERT_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "independent_experts"
ROUTER_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "moe_router"

ROUTER_LR = 1e-4
ROUTER_BATCH_SIZE = 32
ROUTER_EPOCHS = 10


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_expert_paths(seed):
    paths = {}
    for lang in LANGS:
        p = EXPERT_ROOT / f"{lang}_seed{seed}" / "final_model"
        if not p.exists():
            raise FileNotFoundError(f"Expert not found: {p}")
        paths[lang] = str(p)
    return paths


def train_router(seed):
    set_seed(seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"seed={seed}  loading expert models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    moe = MixtureOfExpertsModel(
        backbone_name=BACKBONE,
        expert_model_paths=get_expert_paths(seed),
        router_hidden_dim=256,
        entropy_weight=0.01,
        device=device,
    )

    trainer = RouterTrainer(moe_model=moe)

    train_paths = {lang: str(ROOT / CFG["data"][lang]["train"]) for lang in LANGS}
    val_paths   = {lang: str(ROOT / CFG["data"][lang]["val"])   for lang in LANGS}

    output_dir = ROUTER_ROOT / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    result = trainer.train(
        train_paths=train_paths,
        val_paths=val_paths,
        output_dir=str(output_dir),
        lr=ROUTER_LR,
        batch_size=ROUTER_BATCH_SIZE,
        num_epochs=ROUTER_EPOCHS,
    )

    summary = {
        "seed": seed,
        "expert_paths": get_expert_paths(seed),
        "best_val_loss": result["best_val_loss"],
        "router_path": str(output_dir / "best_router"),
        "history": result["history"],
    }

    with open(output_dir / "router_training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"seed={seed}  best_val_loss={result['best_val_loss']:.4f}  saved to {output_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    args = parser.parse_args()

    ROUTER_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for seed in args.seeds:
        print(f"\nTraining MoE router  seed={seed}")
        all_summaries.append(train_router(seed))

    with open(ROUTER_ROOT / "all_seeds_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\nAll seeds complete.")
    for s in all_summaries:
        print(f"  seed={s['seed']}  best_val_loss={s['best_val_loss']:.4f}")


if __name__ == "__main__":
    main()