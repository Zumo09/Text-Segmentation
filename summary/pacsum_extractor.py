from dataclasses import dataclass
import re
from typing import Callable, List, Optional, Tuple
import random
import pandas as pd
from tqdm import tqdm

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import dot_score

from datasets.fbp_dataset import FBPSummaryDataset
from summary.utils import evaluate_rouge


@dataclass
class HParams:
    beta: float
    lambda1: float
    lambda2: float

    def __repr__(self):
        kws = [f"{key}={value:.2f}" for key, value in self.__dict__.items()]
        return f"{type(self).__name__}({', '.join(kws)})"


class PacSumExtractor:
    def __init__(
        self,
        sentence_transformer: SentenceTransformer,
        extract_num: int = 3,
        hparams: HParams = HParams(0.6, 0.3, 0.3),
    ):

        self.model = sentence_transformer
        self.extract_num = extract_num
        self.hparams = hparams
        self.sep = re.compile("([.?!])")  # keep the separators

    def split(self, document: str) -> List[str]:
        ss = self.sep.split(document)
        # re-concatenate separators to previous sentence
        return [(a + b).strip() for a, b in zip(ss[::2], ss[1::2])]

    def extract_summary(self, text: str) -> str:
        sentences = self.split(text)
        if len(sentences) <= self.extract_num:
            return text

        edge_scores = self.sentence_similarity_matrix(sentences)
        ids = self.select_tops(edge_scores)
        summary = " ".join(map(lambda x: sentences[x], ids))
        return summary

    @torch.no_grad()
    def sentence_similarity_matrix(self, document: List[str]) -> torch.Tensor:
        embeddings: torch.Tensor = self.model.encode(document)  # type: ignore
        return dot_score(embeddings, embeddings)

    def select_tops(
        self,
        edge_scores: torch.Tensor,
        hparams: Optional[HParams] = None,
    ) -> List[int]:
        hparams = hparams or self.hparams

        new_edge_scores = self._normalized_ssm(edge_scores, hparams.beta)
        forward_scores, backward_scores = self._compute_scores(new_edge_scores, 0)

        scores = hparams.lambda1 * forward_scores + hparams.lambda2 * backward_scores

        paired_scores = list(enumerate(scores))
        random.shuffle(paired_scores)  # shuffle to avoid any possible bias
        paired_scores.sort(key=lambda x: float(x[1]), reverse=True)
        extracted = [item[0] for item in paired_scores[: self.extract_num]]

        return extracted

    @staticmethod
    def _normalized_ssm(edge_scores: torch.Tensor, beta: float) -> torch.Tensor:
        min_score = edge_scores.min()
        max_score = edge_scores.max()
        edge_threshold = min_score + beta * (max_score - min_score)
        return edge_scores - edge_threshold

    @staticmethod
    def _compute_scores(
        similarity_matrix: torch.Tensor, edge_threshold: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask = similarity_matrix < edge_threshold
        tri = torch.triu(similarity_matrix, diagonal=1).masked_fill_(mask, 0)
        forward_scores, backward_scores = tri.sum(0), tri.sum(1)
        return forward_scores, backward_scores

    def tune_hparams(
        self,
        data_iterator: FBPSummaryDataset,
        hparams_list: Optional[List[HParams]] = None,
        metrics_keys: Optional[Callable[[pd.DataFrame], float]] = None,
    ) -> HParams:
        metrics_keys = metrics_keys or (lambda df: float(df["rouge-l"]["f"]))
        summaries, references = [], []

        hparams_list = hparams_list or [
            HParams(b, l1, 1 - l1)
            for b in np.linspace(0, 1, 10)
            for l1 in np.linspace(0, 1, 10)
        ]

        for document, info in tqdm(data_iterator):
            assert "summary" in info, "NEED SUMMARY TO TUNE"

            sentences = self.split(document)
            edge_scores = self.sentence_similarity_matrix(sentences)

            tops_list = [self.select_tops(edge_scores, hp) for hp in hparams_list]
            summary_list = [
                " ".join(map(lambda x: sentences[x], ids)) for ids in tops_list
            ]

            summaries.append(summary_list)
            references.append(info["summary"])

        best_rouge = 0
        best_hparam = HParams(np.nan, np.nan, np.nan)
        best_df = None

        for i, hp in enumerate(hparams_list):
            result = evaluate_rouge(
                [s[i] for s in summaries],
                references,
                avg=True,
            )

            metric_value = metrics_keys(result)
            if metric_value > best_rouge:
                best_rouge = metric_value
                best_hparam = hp
                best_df = result
                print(f"Best: {best_rouge:.3f} - {hp}")

        print("-" * 30)
        print(f"The best hyper-parameter:  {best_hparam}")
        print(f"The best rouge score    :  {best_rouge:.3f}")
        print(best_df)

        self.hparams = best_hparam
        return best_hparam
