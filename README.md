# LoRA-MoE Petroleum Translation

Code for the master's thesis experiment **Modular Expert Architectures for Multilingual Domain Adaptation**, focused on multilingual petroleum-domain translation into Norwegian Bokmal.

The project builds language-specific LoRA experts on top of `facebook/nllb-200-distilled-600M`, then trains lightweight routers for modular expert selection. The main experiments cover English, German, French, and Dutch to Norwegian translation.

## Repository Layout

```text
experiments/expert_moe/
  a_train_independent_experts.py      Train one LoRA expert per source language
  a_eval_independent_experts.py       Evaluate independent experts with gold routing
  b_train_multitask_lora.py           Train shared multitask LoRA baseline
  b_eval_multitask_lora.py            Evaluate multitask LoRA baseline
  c_train_sequential_forgetting_lora.py
  c_eval_sequential_forgetting_lora.py
  d_train_sequential_forgetting_full.py
  d_eval_sequential_forgetting_full.py
  e_train_moe_router.py               Train learned MoE router
  e_eval_moe_hard.py                  Evaluate hard-routing MoE
  f_train_langid_router.py            Train LangID-augmented router ablation
  f_eval_langid_router.py             Evaluate LangID-augmented router
  expert_moe.yaml                     Main experiment configuration

scripts/model/
  independent_expert.py               LoRA expert trainer
  mixture_of_experts_model.py         Frozen-backbone LoRA-MoE model
  gated_router.py                     Router network
  router_trainer.py                   Router training utilities

scripts/evaluation/
  base_evaluator.py                   BLEU, chrF, optional COMET
  fta_evaluator.py                    Formal Terminology Accuracy

data/term/
  npd_glossary_multi.json             Petroleum terminology glossary
```

## Data

Large training and evaluation corpora are not committed to this repository. Place data files at the paths expected by `experiments/expert_moe/expert_moe.yaml`:

```text
data/final_splits_npd/
  train.json
  val.json
  test.json

data/gpt_filtered/de_no/
data/gpt_filtered/fr_no/
data/gpt_filtered/nl_no/
  train.json
  val.json
  test.json
```

Each JSON file should contain records with:

```json
{"source": "...", "target": "..."}
```

## Main Experiments

Run commands from the repository root.

Train independent language experts:

```bash
python experiments/expert_moe/a_train_independent_experts.py --langs en de fr nl --seeds 42 123 456
```

Evaluate independent experts:

```bash
python experiments/expert_moe/a_eval_independent_experts.py --seeds 42 123 456
```

Train multitask LoRA baseline:

```bash
python experiments/expert_moe/b_train_multitask_lora.py --seeds 42 123 456
```

Train the MoE router:

```bash
python experiments/expert_moe/e_train_moe_router.py --seeds 42 123 456
```

Evaluate hard-routing MoE:

```bash
python experiments/expert_moe/e_eval_moe_hard.py --seeds 42 123 456
```

COMET can be enabled for evaluation scripts with:

```bash
--use_comet
```

## Outputs

Training checkpoints, final adapters, predictions, routing logs, and metrics are written under:

```text
outputs/expert_moe/
```

This directory is intentionally ignored by Git.

## Notes

- The main MoE setting uses hard top-1 routing at inference.
- Soft routing scripts are exploratory and are not the primary reported system.
- The LangID router scripts are an ablation with explicit source-language identity.
- Experiments were designed for GPU/HPC execution. NLLB-200-distilled-600M and COMET are too heavy for many CPU-only environments.

## Citation

If you use this code, please cite the associated thesis:

```bibtex
@mastersthesis{yang2026modular,
  title  = {Modular Expert Architectures for Multilingual Domain Adaptation: Parameter-Efficient Norwegian Petroleum Translation with LoRA and Gated Routing},
  author = {Yang, Xiaojing},
  school = {Uppsala University},
  year   = {2026}
}
```
