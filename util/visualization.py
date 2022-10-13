import pandas as pd

COLORS = {
    "Position": "\033[31m",  # Red
    "Lead": "\033[32m",
    "Concluding Statement": "\033[33m",  # Yellow
    "Evidence": "\033[34m",  # Blue
    "Claim": "\033[35m",
    "Counterclaim": "\033[36m",
    "Rebuttal": "\033[95m",
    "bold": "\033[1m",
    "end": "\033[0m",
}


def print_segments(id_example: str, text: str, tags: pd.DataFrame):
    tags = tags[tags["id"] == id_example]
    try:
        labels = tags["class"]
    except KeyError:
        labels = tags["discourse_type"]

    indexes = [[int(i) for i in pred.split()] for pred in tags["predictionstring"]]

    boxes = [(i[0], i[-1]) for i in indexes]

    splitted = text.split()

    ents = list(zip(boxes, labels))
    printed = []
    for idx, word in enumerate(splitted):
        found = False
        remaining = ents.copy()
        for ent in ents:
            (s, e), label = ent
            if s <= idx <= e:
                found = True
                print()
                print(COLORS[label] + " ".join(splitted[s: e]) + " [" + label + "]")
                remaining.remove(ent)
                printed.append(ent)

        if not found:
            for ent in printed:
                (s, e), label = ent
                if s <= idx <= e:
                    found = True
                    break

        if not found:
            print(COLORS["end"] + word, end=' ')

        ents = remaining

    if len(ents) != 0:
        raise RuntimeWarning('Not all boxes are printed')