import os
import re
from tqdm import tqdm
from functools import reduce
from typing import Any, Dict, Generator, List, Callable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

import torch
from torch.utils.data import Dataset


class FBPDataset(Dataset):
    def __init__(
        self,
        documents: pd.Series,
        tags: pd.DataFrame,
        encoder: OrdinalEncoder,
        align_target: bool,
    ):
        self.documents = documents
        self.tags = tags
        self.encoder = encoder
        self.align_target = align_target

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index) -> Tuple[str, Dict, Dict]:
        doc_name = self.documents.index[index]
        doc_tags = self.tags[self.tags["id"] == doc_name]  # type: ignore
        tag_cats = (
            torch.Tensor(
                self.encoder.transform(
                    np.array(doc_tags["discourse_type"]).reshape(-1, 1)
                )
            )
            .squeeze()
            .long()
        )

        document = self.documents[doc_name]
        len_sequence = len(document.split())  # type: ignore

        if self.align_target:
            boxes = torch.Tensor(doc_tags[["box_center", "box_length"]].values)
        else:
            boxes = self.map_pred(doc_tags["predictionstring"], len_sequence)

        target = {"labels": tag_cats, "boxes": boxes}
        info = {"id": doc_name, "length": len_sequence}

        return document, target, info  # type: ignore

    @staticmethod
    def map_pred(pred, len_sequence):
        tag_boxes = []
        for p in pred:
            p = p.split()
            p = [int(n) for n in p]
            p = torch.Tensor(p)
            tag_boxes.append([torch.mean(p) / len_sequence, p.size()[0] / len_sequence])

        return torch.Tensor(tag_boxes)


class FBPSummaryDataset(Dataset):
    def __init__(self, documents: pd.Series, summaries: Optional[pd.Series] = None):
        self.documents = documents
        self.summaries = summaries

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, index) -> Tuple[str, Dict[str, Any]]:
        doc_name = self.documents.index[index]
        document: str = self.documents[doc_name]  # type: ignore
        info = {"id": doc_name, "length": len(document.split())}

        if self.summaries is not None and doc_name in self.summaries.index:
            info["summary"] = self.summaries[doc_name]

        return document, info
    
    def __iter__(self) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        for i in range(len(self)):
            yield self[i]


def load_test_texts(path: str, preprocess: List[Callable[[str], str]]) -> pd.Series:
    documents = {}
    for f_name in tqdm(os.listdir(path + "test/"), desc=f"Loading Test Dataset"):
        doc_name = f_name.replace(".txt", "")
        # with open(f_name, 'r') as f:
        with open(path + "test/" + f_name, "r") as f:
            text = reduce(lambda txt, f: f(txt), preprocess, f.read())
            documents[doc_name] = text

    return pd.Series(documents)  # type: ignore


def load_texts(
    path: str, preprocess: List[Callable[[str], str]], dataset_size: float
) -> Tuple[pd.Series, pd.DataFrame]:
    documents = load_only_texts(path + "train/", preprocess, dataset_size)

    types = {"discourse_id": "int64", "discourse_start": int, "discourse_end": int}
    tags = pd.read_csv(os.path.join(path, "train.csv"), dtype=types)

    return pd.Series(documents), tags


def load_only_texts(
    path: str, preprocess: List[Callable[[str], str]], dataset_size: float
) -> pd.Series:
    documents = {}
    listdir = os.listdir(path)

    if dataset_size < 1.0:
        size = int(len(listdir) * dataset_size)
        listdir = listdir[:size]

    for f_name in tqdm(listdir, desc=f"Loading Dataset"):
        doc_name = f_name.replace(".txt", "")
        # with open(f_name, 'r') as f:
        with open(path + f_name, "r") as f:
            text = reduce(lambda txt, f: f(txt), preprocess, f.read())
            documents[doc_name] = text
    return pd.Series(documents)
