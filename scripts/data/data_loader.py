from pathlib import Path
from typing import Dict, Optional, Tuple

from scripts.data.dataset import TranslationDataset

project_root = Path(__file__).resolve().parent.parent.parent

class DataManager:
    """
    Loads train/val/test splits from JSON files.
    Config is a plain dict loaded from config.yaml.
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_dir = project_root / config["paths"]["data_dir"]

    def load_splits(self, use_subset=None, reverse=False):
        train_path = self.data_dir / self.config["data"]["train"]
        val_path = self.data_dir / self.config["data"]["val"]
        test_path = self.data_dir / self.config["data"]["test"]
        src_lang = self.config["model"]["src_lang"]
        tgt_lang = self.config["model"]["tgt_lang"]
        if reverse:
            src_lang, tgt_lang = tgt_lang, src_lang
        train_ds = TranslationDataset.from_json(train_path, src_lang=src_lang, tgt_lang=tgt_lang)
        val_ds = TranslationDataset.from_json(val_path,   src_lang=src_lang, tgt_lang=tgt_lang)
        test_ds = TranslationDataset.from_json(test_path,  src_lang=src_lang, tgt_lang=tgt_lang)
        if reverse:
            for ds in [train_ds, val_ds, test_ds]:
                for sample in ds.samples:
                    sample.source, sample.target = sample.target, sample.source
        if use_subset is not None and use_subset < len(train_ds):
            train_ds = train_ds.subset(use_subset, seed=self.config["project"]["seed"])
        return train_ds, val_ds, test_ds

    def load_splits_for_lang(self, lang: str):
        lang_paths = self.config["data"][lang]
        src_lang = self.config["model"].get("src_lang", lang)
        tgt_lang = self.config["model"]["tgt_lang"]
        train_ds = TranslationDataset.from_json(self.data_dir / lang_paths["train"], src_lang=src_lang, tgt_lang=tgt_lang)
        val_ds = TranslationDataset.from_json(self.data_dir / lang_paths["val"],   src_lang=src_lang, tgt_lang=tgt_lang)
        test_ds = TranslationDataset.from_json(self.data_dir / lang_paths["test"],  src_lang=src_lang, tgt_lang=tgt_lang)
        return train_ds, val_ds, test_ds
