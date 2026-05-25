import sys
import json
import argparse
import random
import gc
import numpy as np
import torch
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

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


def load_mixed_samples(split: str) -> list:
    samples = []
    for lang, cfg in CFG["data"].items():
        path = ROOT / cfg[split]
        if not path.exists():
            print(f"  Skipping {lang} {split}: {path}")
            continue
        with open(path) as f:
            data = json.load(f)
        for item in data:
            samples.append({"source": item["source"], "target": item["target"], "src_lang": lang})
    random.shuffle(samples)
    return samples


def tokenize_samples(samples: list, tokenizer, max_len: int = 128) -> list:
    out = []
    for s in samples:
        tokenizer.src_lang = LANG_CODES[s["src_lang"]]
        tokenizer.tgt_lang = LANG_CODES["no"]
        enc = tokenizer(
            s["source"],
            text_target=s["target"],
            max_length=max_len,
            truncation=True,
            padding=False,
        )
        labels = [(t if t != tokenizer.pad_token_id else -100) for t in enc["labels"]]
        out.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels})
    return out


def train_multitask(seed: int, t: dict) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    output_dir = MODEL_ROOT / f"seed{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)

    print(f"  Loading and tokenizing train split (seed={seed})...")
    train_tokenized = tokenize_samples(load_mixed_samples("train"), tokenizer)
    print(f"  Loading and tokenizing val split...")
    val_tokenized = tokenize_samples(load_mixed_samples("val"), tokenizer)

    train_ds = Dataset.from_list(train_tokenized)
    val_ds = Dataset.from_list(val_tokenized)

    print(f"  Train: {len(train_ds)}  Val: {len(val_ds)}")

    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        BACKBONE,
        dtype=torch.float16,
        attn_implementation="eager",
    )
    base_model = base_model.to("cuda" if torch.cuda.is_available() else "cpu")

    for param in base_model.parameters():
        param.requires_grad = False

    lora_cfg = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=CFG["lora"]["r"],
        lora_alpha=CFG["lora"]["alpha"],
        lora_dropout=CFG["lora"]["dropout"],
        target_modules=CFG["lora"]["target_modules"],
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        seed=seed,
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation"],
        learning_rate=t["lr"],
        warmup_steps=t["warmup_steps"],
        weight_decay=t["weight_decay"],
        lr_scheduler_type="cosine",
        fp16=True,
        predict_with_generate=False,
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        run_name=f"multitask_lora_seed{seed}",
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"])],
    )

    trainer.train()
    final_metrics = trainer.evaluate()

    final_path = output_dir / "final_model"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))

    info = {
        "seed": seed,
        "model_path": str(final_path),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "metrics": final_metrics,
    }
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"  Done seed={seed}  eval_loss={final_metrics.get('eval_loss', 0):.4f}  Model: {final_path}")
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    t_key = "debug" if args.debug else "training"

    for seed in args.seeds:
        print(f"\nTraining multitask LoRA  seed={seed}")
        train_multitask(seed, CFG[t_key])


if __name__ == "__main__":
    main()