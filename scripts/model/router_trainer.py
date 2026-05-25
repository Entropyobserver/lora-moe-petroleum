import gc
import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer

from .mixture_of_experts_model import MixtureOfExpertsModel, LANG_CODES


class MultilingualTranslationDataset(Dataset):

    def __init__(self, samples, tokenizer, max_length=128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, tokenizer, max_length=128):
    from collections import defaultdict

    groups = defaultdict(list)
    for i, s in enumerate(batch):
        groups[s["src_lang"]].append((i, s))

    all_input_ids = [None] * len(batch)
    all_attention_mask = [None] * len(batch)
    all_labels = [None] * len(batch)

    for lang, items in groups.items():
        indices, samples = zip(*items)
        tokenizer.src_lang = LANG_CODES[lang]
        tokenizer.tgt_lang = LANG_CODES["no"]

        enc = tokenizer(
            [s["source"] for s in samples],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        lab = tokenizer(
            text_target=[s["target"] for s in samples],
            max_length=max_length,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        lab.input_ids[lab.input_ids == tokenizer.pad_token_id] = -100

        for j, idx in enumerate(indices):
            all_input_ids[idx] = enc.input_ids[j]
            all_attention_mask[idx] = enc.attention_mask[j]
            all_labels[idx] = lab.input_ids[j]

    pad_id = tokenizer.pad_token_id
    max_src = max(t.size(0) for t in all_input_ids)
    max_tgt = max(t.size(0) for t in all_labels)

    for i in range(len(batch)):
        src_len = all_input_ids[i].size(0)
        if src_len < max_src:
            pad = torch.full((max_src - src_len,), pad_id, dtype=torch.long)
            all_input_ids[i] = torch.cat([all_input_ids[i], pad])
            all_attention_mask[i] = torch.cat([
                all_attention_mask[i],
                torch.zeros(max_src - src_len, dtype=torch.long),
            ])
        tgt_len = all_labels[i].size(0)
        if tgt_len < max_tgt:
            all_labels[i] = torch.cat([all_labels[i], torch.full((max_tgt - tgt_len,), -100, dtype=torch.long)])

    return {
        "input_ids": torch.stack(all_input_ids),
        "attention_mask": torch.stack(all_attention_mask),
        "labels": torch.stack(all_labels),
        "src_langs": [s["src_lang"] for s in batch],
    }


class RouterTrainer:

    def __init__(self, moe_model: MixtureOfExpertsModel):
        self.moe_model = moe_model
        self.tokenizer = moe_model.tokenizer

    def load_multilingual_data(self, data_paths):
        all_samples = []
        for lang, path in data_paths.items():
            with open(path) as f:
                samples = json.load(f)
            for s in samples:
                s["src_lang"] = lang
            all_samples.extend(samples)
        return all_samples

    def train(self, train_paths, val_paths, output_dir, lr=1e-4, batch_size=32, num_epochs=10, run_name="router_training"):
        self.moe_model.unfreeze_router_only()

        train_samples = self.load_multilingual_data(train_paths)
        val_samples = self.load_multilingual_data(val_paths)
        np.random.shuffle(train_samples)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.moe_model.parameters()),
            lr=lr,
            weight_decay=0.01,
        )

        warmup_steps = 200
        steps_per_epoch = max(len(train_samples) // batch_size, 1)
        total_steps = num_epochs * steps_per_epoch
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
        scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])

        best_val_loss = float("inf")
        history = {"train_loss": [], "val_loss": [], "val_routing_acc": []}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.moe_model = self.moe_model.to(device)

        def build_loader(samples):
            return DataLoader(samples, batch_size=batch_size, shuffle=True, collate_fn=lambda b: collate_fn(b, self.tokenizer))

        for epoch in range(num_epochs):
            self.moe_model.train()
            train_loss_sum, train_steps = 0.0, 0

            for batch in build_loader(train_samples):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                out = self.moe_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                out["loss"].backward()
                nn.utils.clip_grad_norm_(self.moe_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_loss_sum += out["loss"].item()
                train_steps += 1

            train_loss = train_loss_sum / max(train_steps, 1)

            self.moe_model.eval()
            val_loss_sum, val_steps = 0.0, 0
            routing_correct, routing_total = 0, 0

            with torch.no_grad():
                for batch in build_loader(val_samples):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["labels"].to(device)

                    out = self.moe_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    val_loss_sum += out["loss"].item()
                    val_steps += 1

                    enc_out = self.moe_model.backbone.get_encoder()(
                        input_ids=input_ids, attention_mask=attention_mask, return_dict=True
                    )
                    weights, _ = self.moe_model.router(enc_out.last_hidden_state)
                    top_experts = weights.argmax(dim=-1)

                    for i, src_lang in enumerate(batch["src_langs"]):
                        gold_idx = self.moe_model.expert_langs.index(src_lang)
                        routing_correct += int(top_experts[i].item() == gold_idx)
                        routing_total += 1

            val_loss = val_loss_sum / max(val_steps, 1)
            val_routing_acc = routing_correct / routing_total if routing_total > 0 else 0.0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_routing_acc"].append(val_routing_acc)

            print(f"Epoch {epoch+1}/{num_epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_routing_acc={val_routing_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.moe_model.save_router(Path(output_dir) / "best_router")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"best_val_loss": best_val_loss, "history": history}