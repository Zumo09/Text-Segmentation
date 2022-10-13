from enum import Enum
from functools import reduce
from typing import Callable, List, Optional, Tuple
import pandas as pd

import torch

from engine import Engine
from models import make_model_tokenizer

from typing import Optional
import numpy as np
import torch

import torch.nn.functional as F
from util import box_ops

from tqdm import tqdm
from datasets.fbp_dataset import FBPSummaryDataset
from summary.utils import evaluate_rouge

from itertools import chain, combinations


class ArgType(Enum):
    Claim = 0
    ConcludingStatement = 1
    Counterclaim = 2
    Evidence = 3
    Lead = 4
    Position = 5
    Rebuttal = 6

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"


class DETRExtractor:
    def __init__(self, args, keep_classes: Optional[List[ArgType]] = None):
        self.device = torch.device(args.device)

        self.model, self.tokenizer = make_model_tokenizer(args.num_classes, None, args)

        self.model.eval()

        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        self.model.load_state_dict(checkpoint["model"])
        self.model.to(self.device)

        self.device = torch.device(args.device)
        self.attention_words = (
            [1219, 6427, 5848, 41752, 766, 679, 959] if args.glob_attn_words else []
        )

        self.keep_classes = keep_classes or [ArgType.ConcludingStatement]

    @property
    def keep_classes(self) -> List[int]:
        return self._keep

    @keep_classes.setter
    def keep_classes(self, argtypes: List[ArgType]) -> None:
        self._keep = [k.value for k in argtypes]

    @torch.no_grad()
    def extract_summary(self, text: str) -> str:
        labels, boxes = self.get_predictions(text)
        splitted = text.split()
        return self._filter_doc(labels, boxes, splitted)

    def _filter_doc(
        self, labels: torch.Tensor, boxes: torch.Tensor, splitted: List[str]
    ) -> str:
        indexes = [
            set(range(start, end))
            for label, (start, end) in zip(labels.squeeze(), boxes.squeeze())
            if label in self.keep_classes
        ]

        if len(indexes) == 0:
            return "[SUMMARY NOT FOUND]"

        filtered = list(reduce(lambda a, b: a.union(b), indexes))
        filtered.sort()

        return " ".join(splitted[i] for i in filtered)

    @torch.no_grad()
    def get_predictions(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        outs = Engine.get_outputs(
            self.tokenizer, self.model, [text], self.device, self.attention_words
        )

        out_logits, out_bbox = outs["pred_logits"].cpu(), outs["pred_boxes"].cpu()
        target_sizes = torch.Tensor([len(text.split())])

        prob = F.softmax(out_logits, -1)
        _, labels = prob.max(-1)

        # from relative [0, 1] to absolute [0, tarx_len] coordinates
        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = out_bbox * scale_fct[:, None, :]
        # and convert to [start, end] format
        boxes = box_ops.box_cl_to_se(boxes)
        # round and positive
        boxes = torch.round(boxes)
        boxes = torch.clamp(boxes, torch.tensor([0]), target_sizes - 1).int()
        return labels, boxes

    def tune_hparams(
        self,
        data_iterator: FBPSummaryDataset,
        keep_classes_hyp: Optional[List[List[ArgType]]] = None,
        metrics_keys: Optional[Callable[[pd.DataFrame], float]] = None,
    ) -> List[ArgType]:
        metrics_keys = metrics_keys or (lambda df: float(df["rouge-l"]["f"]))
        summaries, references = [], []

        keep_classes_hyp = keep_classes_hyp or list(
            chain.from_iterable(
                map(list, combinations([t for t in ArgType], r + 1))
                for r in range(len(ArgType))
            )
        )

        for document, info in tqdm(data_iterator):
            assert "summary" in info, "NEED SUMMARY TO TUNE"

            labels, boxes = self.get_predictions(document)
            splitted = document.split()

            summary_list = []
            for keep in keep_classes_hyp:
                self.keep_classes = keep
                summary_list.append(self._filter_doc(labels, boxes, splitted))

            summaries.append(summary_list)
            references.append(info["summary"])

        best_rouge = 0
        best_hparam = []
        best_df = None

        for i, hp in enumerate(keep_classes_hyp):
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

        self.keep_classes = best_hparam
        return best_hparam
