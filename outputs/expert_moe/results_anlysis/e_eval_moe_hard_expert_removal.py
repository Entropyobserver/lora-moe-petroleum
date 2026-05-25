"""
Expert removal ablation — Hard routing.

For each sentence, the original routing decision is top_expert from routing.json.
To simulate removing expert X, we zero that expert's weight in the continuous
weight vector and take argmax over the remaining three experts.
No model inference needed — operates purely on pre-computed routing.json.
"""

import json
import argparse
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = Path(__file__).parent / "expert_moe.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

LANGS = ["en", "de", "fr", "nl"]
RESULTS_ROOT = ROOT / "outputs" / "expert_moe" / "results"


def removal_accuracy(entries, remove_lang):
    """
    For each entry, zero the removed expert's weight and take argmax.
    Returns accuracy of the new routing decision vs the source language.
    """
    remove_idx = LANGS.index(remove_lang)
    correct = 0

    for e in entries:
        w = np.array(e["weights"])
        if w.ndim == 2:
            w = w[0]
        w = w.copy()
        w[remove_idx] = 0.0
        total = w.sum()
        if total > 0:
            w /= total
        pred = int(np.argmax(w))
        correct += 1  # will be overridden below
        # return pred for counting
        return None  # placeholder

    return None


def simulate_removal(routing_data, src_lang, remove_lang):
    """
    Simulate removing remove_lang expert for all src_lang sentences.
    Returns new routing accuracy (fraction correctly routed to src_lang).
    """
    remove_idx = LANGS.index(remove_lang)
    src_idx = LANGS.index(src_lang)
    entries = routing_data[src_lang]
    correct = 0

    for e in entries:
        w = np.array(e["weights"])
        if w.ndim == 2:
            w = w[0]
        w = w.copy()
        w[remove_idx] = 0.0
        total = w.sum()
        if total > 0:
            w /= total
        pred = int(np.argmax(w))
        if pred == src_idx:
            correct += 1

    return correct / len(entries) if entries else 0.0


def baseline_accuracy(routing_data, src_lang):
    """Routing accuracy from original top_expert field."""
    src_idx = LANGS.index(src_lang)
    entries = routing_data[src_lang]
    correct = sum(1 for e in entries if LANGS.index(e["top_expert"]) == src_idx)
    return correct / len(entries) if entries else 0.0


def run_seed(seed):
    path = RESULTS_ROOT / "moe_hard" / f"seed{seed}" / "routing.json"
    with open(path) as f:
        routing_data = json.load(f)

    results = {}
    for src_lang in LANGS:
        base = baseline_accuracy(routing_data, src_lang)
        results[src_lang] = {"baseline": round(base * 100, 2)}
        for remove_lang in LANGS:
            if remove_lang == src_lang:
                results[src_lang][f"remove_{remove_lang}"] = None
                continue
            new_acc = simulate_removal(routing_data, src_lang, remove_lang)
            delta = (new_acc - base) * 100
            results[src_lang][f"remove_{remove_lang}"] = {
                "accuracy": round(new_acc * 100, 2),
                "delta_pp": round(delta, 2),
            }
            print(f"  seed={seed} src={src_lang} remove={remove_lang}: "
                  f"base={base*100:.1f}% → {new_acc*100:.1f}% (Δ{delta:+.1f}pp)")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    args = parser.parse_args()

    all_results = {}
    for seed in args.seeds:
        print(f"\nExpert removal ablation (hard routing)  seed={seed}")
        all_results[seed] = run_seed(seed)

    # Compute mean delta across seeds
    print("\n=== Mean Δpp across seeds ===")
    print(f"{'src':<6}", end="")
    for remove_lang in LANGS:
        print(f"  remove_{remove_lang.upper():<4}", end="")
    print()

    for src_lang in LANGS:
        print(f"{src_lang.upper():<6}", end="")
        for remove_lang in LANGS:
            if remove_lang == src_lang:
                print(f"  {'N/A':<10}", end="")
                continue
            deltas = [
                all_results[s][src_lang][f"remove_{remove_lang}"]["delta_pp"]
                for s in args.seeds
            ]
            print(f"  {np.mean(deltas):>+6.1f}pp   ", end="")
        print()

    out_path = RESULTS_ROOT / "moe_expert_removal" / "hard_removal_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()