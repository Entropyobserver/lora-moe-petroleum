import sys
import json
import argparse
import random
import gc
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from model.mixture_of_experts_model import MixtureOfExpertsModel
from model.router_trainer import collate_fn

CONFIG_PATH = Path(__file__).parent / "expert_moe.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

BACKBONE = CFG["backbone"]
LANGS = ["en", "de", "fr", "nl"]
EXPERT_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "independent_experts"
OUT_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "langid_router"


class LangAwareRouter(nn.Module):

    def __init__(self, encoder_dim=1024, hidden_dim=256, num_experts=4, entropy_weight=0.01):
        super().__init__()
        self.num_experts = num_experts
        self.entropy_weight = entropy_weight
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

    def entropy_loss(self, weights):
        return -self.entropy_weight * (weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_lang_onehot(src_langs, device):
    idx = torch.tensor([LANGS.index(l) for l in src_langs], dtype=torch.long)
    return F.one_hot(idx, num_classes=len(LANGS)).float().to(device)


def get_expert_paths(seed):
    paths = {}
    for lang in LANGS:
        p = EXPERT_ROOT / f"{lang}_seed{seed}" / "final_model"
        if not p.exists():
            raise FileNotFoundError(f"Expert not found: {p}")
        paths[lang] = str(p)
    return paths


def load_moe(seed, device):
    moe = MixtureOfExpertsModel(
        backbone_name=BACKBONE,
        expert_model_paths=get_expert_paths(seed),
        router_hidden_dim=256,
        entropy_weight=0.01,
        device=device,
    )
    moe.freeze_experts()
    return moe


def forward_step(moe, router, input_ids, attention_mask, labels, src_langs, device):
    with torch.no_grad():
        enc = moe.backbone.get_encoder()(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
    lang_oh = make_lang_onehot(src_langs, device)
    weights, _ = router(enc.last_hidden_state, lang_oh)

    mixed = None
    for i, lang in enumerate(LANGS):
        moe.peft_model.set_adapter(lang)
        with torch.no_grad():
            out = moe.peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        w = weights[:, i].unsqueeze(-1).unsqueeze(-1)
        mixed = w * out.logits if mixed is None else mixed + w * out.logits

    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)(mixed.view(-1, mixed.size(-1)), labels.view(-1))
    return ce_loss + router.entropy_loss(weights), weights


def routing_accuracy(moe, router, loader, device):
    moe.eval()
    router.eval()
    correct = {l: 0 for l in LANGS}
    total = {l: 0 for l in LANGS}

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            enc = moe.backbone.get_encoder()(input_ids=ids, attention_mask=mask, return_dict=True)
            lang_oh = make_lang_onehot(batch["src_langs"], device)
            weights, _ = router(enc.last_hidden_state, lang_oh)
            preds = weights.argmax(dim=-1)
            for i, src in enumerate(batch["src_langs"]):
                total[src] += 1
                correct[src] += int(preds[i].item() == LANGS.index(src))

    per_lang = {l: correct[l] / total[l] if total[l] > 0 else 0.0 for l in LANGS}
    overall = sum(correct.values()) / sum(total.values())
    return per_lang, overall


def load_samples(split):
    samples = []
    for lang in LANGS:
        with open(ROOT / CFG["data"][lang][split]) as f:
            for s in json.load(f):
                s["src_lang"] = lang
                samples.append(s)
    return samples


def train(seed, lr, batch_size, epochs):
    set_seed(seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # expert_seed = seed: each router seed loads the same-seed expert
    moe = load_moe(seed, device).to(device)
    router = LangAwareRouter(encoder_dim=moe.backbone.config.hidden_size).to(device)

    optimizer = optim.AdamW(router.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_samples = load_samples("train")
    val_samples = load_samples("val")

    def make_loader(samples, shuffle=True):
        return DataLoader(
            samples, batch_size=batch_size, shuffle=shuffle,
            collate_fn=lambda b: collate_fn(b, moe.tokenizer)
        )

    val_loader = make_loader(val_samples, shuffle=False)
    out_dir = OUT_ROOT / f"seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        moe.eval()
        router.train()
        np.random.shuffle(train_samples)

        train_loss_sum, steps = 0.0, 0
        for batch in make_loader(train_samples):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labs = batch["labels"].to(device)

            optimizer.zero_grad()
            loss, _ = forward_step(moe, router, ids, mask, labs, batch["src_langs"], device)
            loss.backward()
            nn.utils.clip_grad_norm_(router.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            steps += 1

        scheduler.step()

        val_loss_sum, val_steps = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                labs = batch["labels"].to(device)
                loss, _ = forward_step(moe, router, ids, mask, labs, batch["src_langs"], device)
                val_loss_sum += loss.item()
                val_steps += 1

        val_loss = val_loss_sum / val_steps
        per_lang, overall = routing_accuracy(moe, router, val_loader, device)

        print(
            f"[seed={seed} epoch={epoch+1}/{epochs}]"
            f"  train={train_loss_sum/steps:.4f}"
            f"  val={val_loss:.4f}"
            f"  acc={overall:.4f}  "
            + "  ".join(f"{l}={per_lang[l]*100:.1f}%" for l in LANGS)
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(router.state_dict(), out_dir / "best_router.pt")
            torch.save(
                {"encoder_dim": moe.backbone.config.hidden_size, "hidden_dim": 256, "num_experts": 4},
                out_dir / "router_config.pt",
            )

    per_lang, overall = routing_accuracy(moe, router, val_loader, device)
    result = {
        "seed": seed,
        "expert_seed": seed,
        "best_val_loss": best_val_loss,
        "overall_acc": overall,
        "per_lang_acc": per_lang,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  LangID-augmented router   seed={seed}   expert_seed={seed}")
        print(f"{'='*60}")
        results.append(train(seed, args.lr, args.batch_size, args.epochs))

    print(f"\n{'='*60}")
    print("Routing accuracy summary (val set)")
    for r in results:
        print(f"  seed={r['seed']}  overall={r['overall_acc']*100:.1f}%  " +
              "  ".join(f"{l}={r['per_lang_acc'][l]*100:.1f}%" for l in LANGS))

    with open(OUT_ROOT / "all_seeds_summary.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()