from typing import Sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder


def get_class_freqs(
    indexes: Sequence, tags: pd.DataFrame, encoder: OrdinalEncoder, num_queries: int
) -> np.ndarray:
    data = tags[tags["id"].isin(indexes)]

    no_obj = (num_queries - data.groupby(by="id")["discourse_id"].count()).sum()
    types = data.groupby(by="discourse_type")["discourse_id"].count()

    counts = np.zeros(len(encoder.categories_[0]) + 1)
    counts[-1] = no_obj
    for cat in data["discourse_type"].unique():
        idx = int(encoder.transform([[cat]])[0, 0])
        counts[idx] = types[cat]

    if (counts == 0).any():
        raise RuntimeError("Category not included in training set... Increase dataset size!")

    return counts / counts.sum()
