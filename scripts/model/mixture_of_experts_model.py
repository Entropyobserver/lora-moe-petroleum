import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model, TaskType

from .gated_router import GatedRouter


EXPERT_LANGS = ["en", "de", "fr", "nl"]

LANG_CODES = {
    "en": "eng_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "nl": "nld_Latn",
    "no": "nob_Latn",
}


class MixtureOfExpertsModel(nn.Module):

    def __init__(
        self,
        backbone_name: str = "facebook/nllb-200-distilled-600M",
        expert_model_paths: Optional[Dict[str, str]] = None,
        router_hidden_dim: int = 256,
        entropy_weight: float = 0.01,
        device: str = "cuda",
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.device = device
        self.expert_langs = EXPERT_LANGS

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_name)

        self.backbone = AutoModelForSeq2SeqLM.from_pretrained(
            backbone_name,
            dtype=torch.float16,
            attn_implementation="eager",
        ).to(device)
        for param in self.backbone.parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],
            bias="none",
        )

        if expert_model_paths:
            first_lang = list(expert_model_paths.keys())[0]
            self.peft_model = PeftModel.from_pretrained(
                self.backbone, expert_model_paths[first_lang], adapter_name=first_lang
            )
            for lang, path in expert_model_paths.items():
                if lang == first_lang:
                    continue
                self.peft_model.load_adapter(path, adapter_name=lang)
        else:
            self.peft_model = get_peft_model(
                self.backbone, lora_config, adapter_name=self.expert_langs[0]
            )
            for lang in self.expert_langs[1:]:
                self.peft_model.add_adapter(lang, lora_config)

        self.expert_langs = [
            l for l in self.expert_langs
            if l in (list(expert_model_paths.keys()) if expert_model_paths else self.expert_langs)
        ]

        encoder_hidden_size = self.backbone.config.hidden_size
        self.router = GatedRouter(
            input_dim=encoder_hidden_size,
            hidden_dim=router_hidden_dim,
            num_experts=len(self.expert_langs),
            entropy_weight=entropy_weight,
        )

    def forward(self, input_ids, attention_mask, labels=None):
        with torch.no_grad():
            encoder_outputs = self.backbone.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        hidden_states = encoder_outputs.last_hidden_state
        router_weights, router_logits = self.router(hidden_states)

        mixed_logits = None
        for idx, lang in enumerate(self.expert_langs):
            self.peft_model.set_adapter(lang)
            with torch.no_grad():
                out = self.peft_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            weighted_logits = out.logits * router_weights[:, idx].unsqueeze(-1).unsqueeze(-1)
            mixed_logits = weighted_logits if mixed_logits is None else (mixed_logits + weighted_logits)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(mixed_logits.view(-1, mixed_logits.size(-1)), labels.view(-1))
            loss = loss + self.router.entropy_loss(router_weights)

        return {"loss": loss, "logits": mixed_logits, "router_weights": router_weights}

    def generate(self, input_ids, attention_mask, tgt_lang_code, inference_mode="hard", forced_weights=None, **kwargs):
        forced_bos = self.tokenizer.convert_tokens_to_ids(tgt_lang_code)

        with torch.no_grad():
            encoder_outputs = self.backbone.get_encoder()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
        router_weights, _ = self.router(encoder_outputs.last_hidden_state)

        if inference_mode == "hard":
            top_expert_lang = self.expert_langs[router_weights.mean(dim=0).argmax().item()]
            self.peft_model.set_adapter(top_expert_lang)
            return self.peft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=forced_bos,
                **kwargs,
            )

        weights = forced_weights if forced_weights is not None else \
            router_weights.mean(dim=0).cpu().float().tolist()

        try:
            self.peft_model.base_model.delete_adapter("soft_merged")
        except Exception:
            pass

        self.peft_model.base_model.add_weighted_adapter(
            adapters=self.expert_langs,
            weights=weights,
            adapter_name="soft_merged",
            combination_type="linear",
        )
        self.peft_model.set_adapter("soft_merged")

        try:
            return self.peft_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                forced_bos_token_id=forced_bos,
                **kwargs,
            )
        finally:
            try:
                self.peft_model.base_model.delete_adapter("soft_merged")
            except Exception:
                pass

    def freeze_experts(self):
        for name, param in self.peft_model.named_parameters():
            if "lora" in name:
                param.requires_grad = False

    def unfreeze_router_only(self):
        self.freeze_experts()
        for param in self.router.parameters():
            param.requires_grad = True

    def save_router(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        torch.save(self.router.state_dict(), Path(output_dir) / "router.pt")

    def load_router(self, router_path):
        self.router.load_state_dict(torch.load(router_path, map_location=self.device))
