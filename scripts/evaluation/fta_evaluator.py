import json
from typing import Dict, List, Optional, Tuple

from .base_evaluator import BaseEvaluator


LANG_KEY_MAP = {
    "en": None,
    "de": "de",
    "fr": "fr",
    "nl": "nl",
    "no": "no",
}


class FTAEvaluator(BaseEvaluator):
    def __init__(
        self,
        glossary_path: str,
        src_lang: str = "en",
        tgt_lang: str = "no",
        use_comet: bool = False,
    ):
        super().__init__(use_comet=use_comet)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        with open(glossary_path, encoding="utf-8") as f:
            glossary = json.load(f)

        src_key = LANG_KEY_MAP.get(src_lang)
        tgt_key = tgt_lang

        self.term_pairs: List[Tuple[str, str]] = []
        for en_term, translations in glossary.items():
            source_term = en_term if src_key is None else translations.get(src_key, en_term)
            target_term = translations.get(tgt_key, "")

            if source_term and target_term:
                self.term_pairs.append((source_term.lower(), target_term.lower()))

    def _terms_in(self, source: str) -> List[Tuple[str, str]]:
        source_lower = source.lower()
        return [
            (source_term, target_term)
            for source_term, target_term in self.term_pairs
            if source_term in source_lower
        ]

    def compute_fta_single(
        self,
        source: str,
        prediction: str,
    ) -> Optional[float]:
        terms = self._terms_in(source)
        if not terms:
            return None

        prediction_lower = prediction.lower()
        hits = sum(1 for _, target_term in terms if target_term in prediction_lower)
        return hits / len(terms)

    def compute_fta(
        self,
        sources: List[str],
        predictions: List[str],
    ) -> Dict[str, float]:
        scores = []
        total_hits = 0
        total_terms = 0

        for source, prediction in zip(sources, predictions):
            terms = self._terms_in(source)
            if not terms:
                continue

            prediction_lower = prediction.lower()
            hits = sum(1 for _, target_term in terms if target_term in prediction_lower)

            total_hits += hits
            total_terms += len(terms)
            scores.append(hits / len(terms))

        if not scores:
            return {
                "fta": 0.0,
                "fta_mean_sentence": 0.0,
                "fta_coverage": 0.0,
                "fta_sentences": 0,
                "fta_terms_total": 0,
            }

        return {
            "fta": total_hits / total_terms,
            "fta_mean_sentence": sum(scores) / len(scores),
            "fta_coverage": len(scores) / len(sources),
            "fta_sentences": len(scores),
            "fta_terms_total": total_terms,
        }

    def evaluate_all(
        self,
        sources: List[str],
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        metrics = super().evaluate_all(sources, predictions, references)
        metrics.update(self.compute_fta(sources, predictions))
        return metrics