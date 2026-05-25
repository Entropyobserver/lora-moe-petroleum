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

import evaluate as hf_evaluate
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

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
MODEL_ROOT = ROOT / "outputs" / "expert_moe" / "models" / "sequential_forgetting"

BLEU_METRIC = hf_evaluate.load("bleu")
CHRF_METRIC = hf_evaluate.load("chrf")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def make_dataset(data, tokenizer, src_lang):
    tokenizer.src_lang = LANG_CODES[src_lang]
    tokenizer.tgt_lang = LANG_CODES["no"]
    rows = []
    for item in data:
        enc = tokenizer(item["source"], text_target=item["target"], max_length=128, truncation=True, padding=False)
        labels = [t if t != tokenizer.pad_token_id else -100 for t in enc["labels"]]
        rows.append({"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels})
    return Dataset.from_list(rows)


def evaluate_lang_bleu(model, tokenizer, lang):
    """Translate the validation set for a given source language and return BLEU and chrF.
    The validation set is used here to track the forgetting trajectory during training;
    the test set is reserved for the final evaluation script only.
    """
    data = load_json(ROOT / CFG["data"][lang]["val"])
    device = next(model.parameters()).device
    tokenizer.src_lang = LANG_CODES[lang]
    forced_bos = tokenizer.convert_tokens_to_ids(LANG_CODES["no"])
    preds = []
    for sample in data:
        inputs = tokenizer(sample["source"], return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = model.generate(**inputs, forced_bos_token_id=forced_bos, num_beams=5, max_length=128)
        preds.append(tokenizer.decode(out[0], skip_special_tokens=True))
    refs = [x["target"] for x in data]
    bleu = BLEU_METRIC.compute(predictions=preds, references=[[r] for r in refs])["bleu"]
    chrf = CHRF_METRIC.compute(predictions=preds, references=refs)["score"]
    return {"bleu": bleu, "chrf": chrf}


def load_base_model():
    base = AutoModelForSeq2SeqLM.from_pretrained(
        BACKBONE,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    base = base.to("cuda" if torch.cuda.is_available() else "cpu")
    for p in base.parameters():
        p.requires_grad = False
    return base


def train_one_phase(model, tokenizer, src_lang, step_dir, t, seed):
    train_ds = make_dataset(load_json(ROOT / CFG["data"][src_lang]["train"]), tokenizer, src_lang)
    val_ds = make_dataset(load_json(ROOT / CFG["data"][src_lang]["val"]), tokenizer, src_lang)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if preds.ndim == 3:
            preds = np.argmax(preds, axis=-1)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        bleu = BLEU_METRIC.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        chrf = CHRF_METRIC.compute(predictions=decoded_preds, references=decoded_labels)
        return {"bleu": bleu["bleu"], "chrf": chrf["score"]}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(step_dir / "checkpoints"),
        num_train_epochs=t["num_epochs"],
        per_device_train_batch_size=t["batch_size"],
        per_device_eval_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation"],
        learning_rate=t["lr"],
        warmup_steps=t["warmup_steps"],
        lr_scheduler_type="cosine",
        weight_decay=t["weight_decay"],
        fp16=False,
        bf16=True,
        predict_with_generate=True,
        eval_strategy="steps",
        eval_steps=t["eval_steps"],
        save_strategy="steps",
        save_steps=t["eval_steps"],
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        logging_steps=50,
        report_to="none",
        seed=seed,
        data_seed=seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"])],
    )

    trainer.train()
    save_path = step_dir / "final_model"
    save_path.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(save_path))
    return str(save_path)


def run_seed(seed, t):
    set_seed(seed)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    seed_dir = MODEL_ROOT / f"seed{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BACKBONE)
    trajectory = []
    prev_adapter_path = None

    for step_idx, lang in enumerate(TRAIN_ORDER):
        print(f"seed={seed}  step {step_idx+1}/{len(TRAIN_ORDER)}  training {lang.upper()}")

        step_dir = seed_dir / f"step{step_idx+1}_{lang}"
        step_dir.mkdir(parents=True, exist_ok=True)

        torch.manual_seed(seed)
        base = load_base_model()

        if prev_adapter_path is None:
            lora_cfg = LoraConfig(
                task_type=TaskType.SEQ_2_SEQ_LM,
                r=CFG["lora"]["r"],
                lora_alpha=CFG["lora"]["alpha"],
                lora_dropout=CFG["lora"]["dropout"],
                target_modules=CFG["lora"]["target_modules"],
                bias="none",
            )
            model = get_peft_model(base, lora_cfg)
        else:
            model = PeftModel.from_pretrained(base, prev_adapter_path)
            for name, p in model.named_parameters():
                if "lora" in name:
                    p.requires_grad = True

        saved_path = train_one_phase(model, tokenizer, lang, step_dir, t, seed)

        del model, base
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        torch.manual_seed(seed)
        base_eval = load_base_model()
        eval_model = PeftModel.from_pretrained(base_eval, saved_path)
        eval_model.eval()

        # EN--NO is evaluated at every step to track the forgetting trajectory.
        en_metrics = evaluate_lang_bleu(eval_model, tokenizer, "en")
        cur_metrics = evaluate_lang_bleu(eval_model, tokenizer, lang)

        step_record = {
            "step": step_idx + 1,
            "trained_on": lang,
            "model_path": saved_path,
            "en_bleu": en_metrics["bleu"],
            "en_chrf": en_metrics["chrf"],
            f"{lang}_bleu": cur_metrics["bleu"],
            f"{lang}_chrf": cur_metrics["chrf"],
        }

        # At the final step, evaluate all four language directions to align with
        # the full fine-tuning baseline and give a complete picture of forgetting.
        if step_idx == len(TRAIN_ORDER) - 1:
            print("  Final step: evaluating all language directions...")
            for eval_lang in TRAIN_ORDER:
                if eval_lang == lang:
                    continue
                extra = evaluate_lang_bleu(eval_model, tokenizer, eval_lang)
                step_record[f"{eval_lang}_final_bleu"] = extra["bleu"]
                step_record[f"{eval_lang}_final_chrf"] = extra["chrf"]
                print(f"    {eval_lang.upper()}->NO  BLEU: {extra['bleu']:.4f}  chrF: {extra['chrf']:.2f}")

        del eval_model, base_eval
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trajectory.append(step_record)
        print(f"  EN BLEU after {lang.upper()}: {en_metrics['bleu']:.4f}  |  {lang.upper()} BLEU: {cur_metrics['bleu']:.4f}")
        prev_adapter_path = saved_path

    delta_forget_bleu = trajectory[0]["en_bleu"] - trajectory[-1]["en_bleu"]
    delta_forget_chrf = trajectory[0]["en_chrf"] - trajectory[-1]["en_chrf"]
    summary = {
        "seed": seed,
        "train_order": TRAIN_ORDER,
        "trajectory": trajectory,
        "delta_forget_bleu": delta_forget_bleu,
        "delta_forget_chrf": delta_forget_chrf,
        "en_bleu_after_en": trajectory[0]["en_bleu"],
        "en_bleu_after_fr": trajectory[-1]["en_bleu"],
        "en_chrf_after_en": trajectory[0]["en_chrf"],
        "en_chrf_after_fr": trajectory[-1]["en_chrf"],
    }

    with open(seed_dir / "forgetting_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"seed={seed}  delta_forget_bleu={delta_forget_bleu:.4f}  ({trajectory[0]['en_bleu']:.4f} -> {trajectory[-1]['en_bleu']:.4f})  delta_forget_chrf={delta_forget_chrf:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=CFG["training"]["seeds"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    t_key = "debug" if args.debug else "training"

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    all_summaries = []

    for seed in args.seeds:
        print(f"\nSeed {seed}")
        all_summaries.append(run_seed(seed, CFG[t_key]))

    with open(MODEL_ROOT / "all_seeds_summary.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("All seeds complete.")
    for s in all_summaries:
        print(f"  seed={s['seed']}  delta_forget_bleu={s['delta_forget_bleu']:.4f}  delta_forget_chrf={s['delta_forget_chrf']:.4f}")


if __name__ == "__main__":
    main()