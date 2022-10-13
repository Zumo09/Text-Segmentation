from typing import List, Optional, Protocol, Tuple
from rouge import Rouge

import pandas as pd
from tqdm import tqdm
from datasets.fbp_dataset import FBPSummaryDataset


class Extractor(Protocol):
    def extract_summary(self, text: str) -> str:
        ...


def evaluate_rouge(
    summaries: List[str], references: List[str], avg: bool = True
) -> pd.DataFrame:
    rouge = Rouge()
    res = rouge.get_scores(summaries, references, avg=avg)
    scores = pd.DataFrame(res)
    scores["mean"] = scores.mean(1)
    return scores


def extract_summary_iterator(
    extractor: Extractor,
    data_iterator: FBPSummaryDataset,
) -> Tuple[List[str], Optional[pd.DataFrame]]:

    summaries = []
    references = []

    for sentences, info in tqdm(data_iterator):
        if "summary" in info.keys():
            references.append(info["summary"])

        summary = extractor.extract_summary(sentences)
        summaries.append(summary)

    if len(summaries) == len(references):
        return summaries, evaluate_rouge(summaries, references, avg=True)
    return summaries, None
