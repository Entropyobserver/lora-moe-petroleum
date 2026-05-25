import gc
import json
from pathlib import Path
from typing import Dict, List, Optional

import evaluate
import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

# Flash attention was unstable for this setup on the cluster.
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


LANG_CODES = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "nl": "nld_Latn",
    "no": "nob_Latn",
}


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ExpertTrainer:
    def __init__(
        self,
        backbone_name: str = "facebook/nllb-200-distilled-600M",
        src_lang: str = "en",
        tgt_lang: str = "no",
        use_comet: bool = False,
    ):
        self.backbone_name = backbone_name
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.src_lang_code = LANG_CODES[src_lang]
        self.tgt_lang_code = LANG_CODES[tgt_lang]

        self.use_comet = use_comet
        self.val_sources: Optional[List[str]] = None

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)
        self.bleu = evaluate.load("bleu")
        self.chrf = evaluate.load("chrf")

        self.comet_model = None
        if self.use_comet:
            try:
                from comet import download_model, load_from_checkpoint

                comet_path = download_model("Unbabel/wmt22-comet-da")
                self.comet_model = load_from_checkpoint(comet_path)
                print("COMET model loaded successfully")
            except Exception as e:
                print(f"COMET loading failed: {e}. Continuing without COMET.")
                self.use_comet = False

    def setup_model(self, r: int = 16, alpha: int = 32, dropout: float = 0.1):
        base_model = AutoModelForSeq2SeqLM.from_pretrained(
            self.backbone_name,
            dtype=torch.float16,
            attn_implementation="eager",
        )
        base_model = base_model.to("cuda" if torch.cuda.is_available() else "cpu")

        for param in base_model.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            bias="none",
        )

        model = get_peft_model(base_model, lora_config)
        model.print_trainable_parameters()
        return model

    def load_data(self, path: str) -> List[Dict]:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def tokenize(self, examples: Dict) -> Dict:
        self.tokenizer.src_lang = self.src_lang_code
        model_inputs = self.tokenizer(
            examples["source"],
            max_length=128,
            truncation=True,
            padding=False,
        )

        self.tokenizer.tgt_lang = self.tgt_lang_code
        labels = self.tokenizer(
            text_target=examples["target"],
            max_length=128,
            truncation=True,
            padding=False,
        )

        model_inputs["labels"] = [
            [(token if token != self.tokenizer.pad_token_id else -100) for token in label]
            for label in labels["input_ids"]
        ]
        return model_inputs

    def compute_metrics(self, eval_pred) -> Dict:
        predictions, labels = eval_pred

        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        decoded_preds = self.tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(
            labels,
            skip_special_tokens=True,
        )

        bleu = self.bleu.compute(
            predictions=decoded_preds,
            references=[[ref] for ref in decoded_labels],
        )
        chrf = self.chrf.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )

        metrics = {
            "bleu": bleu["bleu"],
            "chrf": chrf["score"],
        }

        if self.use_comet and self.comet_model and self.val_sources is not None:
            try:
                comet_data = [
                    {"src": src, "mt": pred, "ref": ref}
                    for src, pred, ref in zip(
                        self.val_sources[: len(decoded_preds)],
                        decoded_preds,
                        decoded_labels,
                    )
                ]
                result = self.comet_model.predict(comet_data, batch_size=8, gpus=1)
                metrics["comet"] = result["system_score"]
            except Exception as e:
                print(f"COMET calculation failed: {e}")

        return metrics

    def train(
        self,
        train_path: str,
        val_path: str,
        output_dir: str,
        lr: float = 5e-4,
        batch_size: int = 8,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 5,
        warmup_steps: int = 200,
        eval_steps: int = 200,
        early_stopping_patience: int = 3,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.1,
        weight_decay: float = 0.01,
        seed: int = 42,
        run_name: str = None,
    ) -> Dict:
        clear_memory()

        model = self.setup_model(r=r, alpha=alpha, dropout=dropout)

        forced_bos_id = self.tokenizer.convert_tokens_to_ids(self.tgt_lang_code)
        model.config.forced_bos_token_id = forced_bos_id
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.forced_bos_token_id = forced_bos_id

        train_data = self.load_data(train_path)
        val_data = self.load_data(val_path)

        if self.use_comet:
            self.val_sources = [item["source"] for item in val_data]

        train_dataset = Dataset.from_list(train_data).map(self.tokenize, batched=True)
        val_dataset = Dataset.from_list(val_data).map(self.tokenize, batched=True)

        collator = DataCollatorForSeq2Seq(self.tokenizer, model=model, padding=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            seed=seed,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=lr,
            warmup_steps=warmup_steps,
            lr_scheduler_type="cosine",
            weight_decay=weight_decay,
            fp16=True,
            predict_with_generate=True,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            save_steps=eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="bleu",
            greater_is_better=True,
            logging_steps=50,
            run_name=run_name or f"{self.src_lang}-{self.tgt_lang}-expert",
            report_to="none",
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=collator,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=early_stopping_patience,
                )
            ],
        )

        trainer.train()
        metrics = trainer.evaluate()

        final_path = Path(output_dir) / "final_model"
        final_path.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(final_path))

        clear_memory()

        return {
            "metrics": metrics,
            "model_path": str(final_path),
        }