from typing import Optional
from sklearn.preprocessing import OrdinalEncoder
import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np

from util import box_ops


class FBPPostProcess:
    """This module converts the model's output into the format expected by the Kaggle api"""

    def __init__(
        self, encoder: OrdinalEncoder, tags: pd.DataFrame, no_obj_class: int
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.tags = tags
        self.no_obj_class = no_obj_class
        self.reset_results()

    @property
    def results(self) -> pd.DataFrame:
        """The DataFrame to be submitted for the challenge"""
        if len(self._results) == 0:
            return pd.DataFrame(columns=["id", "class", "predictionstring", "score"])
        return pd.DataFrame(self._results)

    def reset_results(self):
        self._results = []

    @staticmethod
    def _predstr_to_set(pred: str):
        return set(int(i) for i in pred.split())

    @staticmethod
    def prec_rec_f1(tp, fp, fn):
        prec = tp / (tp + fp + 1e-3)
        recall = tp / (tp + fn + 1e-3)
        f1 = tp / (tp + 0.5 * (fp + fn) + 1e-3)
        return {"precision": prec, "recall": recall, "f1": f1}

    def _evaluate_doc_class(self, pred, tags):
        lp = len(pred)
        lt = len(tags)
        overlaps_pt = np.zeros(shape=(lp, lt))
        overlaps_tp = np.zeros(shape=(lp, lt))
        p_sets = [self._predstr_to_set(ps) for ps in pred["predictionstring"]]
        t_sets = [self._predstr_to_set(ps) for ps in tags["predictionstring"]]

        for p in range(lp):
            p_set = p_sets[p]
            for t in range(lt):
                t_set = t_sets[t]
                overlaps_pt[p, t] = len(p_set.intersection(t_set)) / len(t_set)
                overlaps_tp[p, t] = len(t_set.intersection(p_set)) / len(p_set)

        tp = 0
        fn = 0
        pred_matched = []
        for t in range(lt):
            larger = 0
            selected = None
            for p in range(lp):
                if p in pred_matched:
                    continue
                
                opt = overlaps_pt[p, t]
                otp = overlaps_tp[p, t]
                if opt >= 0.5 and otp >= 0.5:
                    if (opt + otp) > larger:
                        larger = opt + otp
                        selected = p
                
            if selected is None:
                fn += 1
            else:
                pred_matched.append(selected)
                tp += 1

        fp = lp - tp
        return tp, fp, fn

    def evaluate(self, results: Optional[pd.DataFrame] = None):
        """
        Evaluation metric defined by the Kaggle Challenge
        """
        if results is None:
            results = self.results
        gb_res = results.groupby(by="id")
        gb_tag = self.tags.groupby(by="id")

        report = {}
        for cls in self.tags["discourse_type"].unique():
            tp, fp, fn = 0, 0, 0
            for doc_id in results["id"].unique():
                pred = gb_res.get_group(doc_id)
                tags = gb_tag.get_group(doc_id)

                pred = pred[pred["class"] == cls]
                tags = tags[tags["discourse_type"] == cls]
                a, b, c = self._evaluate_doc_class(pred, tags)
                tp += a
                fp += b
                fn += c
            report[cls] = self.prec_rec_f1(tp, fp, fn)

        report["macro_avg"] = {
            "precision": sum(cls_rep["precision"] for cls_rep in report.values())
            / len(report),
            "recall": sum(cls_rep["recall"] for cls_rep in report.values())
            / len(report),
            "f1": sum(cls_rep["f1"] for cls_rep in report.values()) / len(report),
        }
        return pd.DataFrame(report).transpose()

    @torch.no_grad()
    def add_outputs(self, outputs, infos):
        """Format the outputs and save them to a dataframe
        Parameters:
            outputs: raw outputs of the model
            infos: list of dictionaries of length [batch_size] containing the length of each document of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs["pred_logits"].cpu(), outputs["pred_boxes"].cpu()
        target_sizes = torch.Tensor([info["length"] for info in infos])

        assert len(out_logits) == len(target_sizes)

        prob = F.softmax(out_logits, -1)
        scores, labels = prob.max(-1)

        # from relative [0, 1] to absolute [0, tarx_len] coordinates
        scale_fct = torch.stack([target_sizes, target_sizes], dim=1)
        boxes = out_bbox * scale_fct[:, None, :]
        # and convert to [start, end] format
        boxes = box_ops.box_cl_to_se(boxes)
        # round and positive
        boxes = torch.round(boxes)
        boxes = torch.relu(boxes).int()

        for i, l, s, b in zip(infos, labels, scores, boxes):
            self._add(i, l, s, b)

    def _add(self, info, labels, scores, boxes):
        doc_id = info["id"]
        for l, s, b in zip(labels, scores, boxes):
            if l != self.no_obj_class:
                l = self.encoder.inverse_transform(l.reshape(-1, 1))
                self._results.append(
                    {
                        "id": doc_id,
                        "class": l[0][0],
                        "predictionstring": self.prediction_string(b),
                        "score": s.item(),
                    }
                )

    @staticmethod
    def prediction_string(box):
        start, end = box
        return " ".join(str(i) for i in range(start, end + 1))
