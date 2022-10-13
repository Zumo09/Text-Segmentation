import pandas as pd
import math


class BaselineClassifier:
    def __init__(
        self,
        major_class: str,
        median_length: int,
    ):
        self.major_class = major_class
        self.median_length = median_length

    def predict(self, id: str, text: str):
        pred_table = pd.DataFrame(columns=["id", "class", "predictionstring", "score"])
        text_splitted = (
            text.replace(",", " ").replace(".", " ").replace("'", " ").split()
        )
        for i in range(math.ceil(len(text_splitted) / self.median_length)):
            pred = {}
            pred["id"] = id
            pred["class"] = self.major_class
            pred["score"] = 1.0
            str_prediction = ""
            for k in range(
                self.median_length * i,
                min(self.median_length * (i + 1), len(text_splitted)),
            ):
                str_prediction += str(f" {k}")
            pred["predictionstring"] = [str_prediction]
            # print(pred)
            pred_table = pd.concat([pred_table, pd.DataFrame(pred)])
        return pred_table

    def dataset_prediction(self, dataset):
        pred_table = pd.DataFrame(columns=["id", "class", "predictionstring", "score"])
        for idx in dataset.documents.index:
            text = dataset.documents[idx]
            pred = self.predict(idx, text)
            pred_table = pd.concat([pred_table, pred])

        pred_table = pred_table.reset_index().drop("index", axis=1)
        return pred_table
