import re
import unicodedata


REPLACE_BY_SPACE_RE = re.compile("[/(){}\[\]\|@_-]")
GOOD_SYMBOLS_RE = re.compile("[^0-9a-z .,;!?']")


def normalize(text: str) -> str:
    """
    Delete strange charachters such as \xa0 due to the format
    """
    return unicodedata.normalize("NFKD", text)


def lower(text: str) -> str:
    return text.lower()


def replace_special_characters(text: str) -> str:
    return REPLACE_BY_SPACE_RE.sub(" ", text)


def filter_out_uncommon_symbols(text: str) -> str:
    return GOOD_SYMBOLS_RE.sub("", text)


def strip_text(text: str) -> str:
    return text.strip()


PIPELINE = [
    normalize,
    lower,
    replace_special_characters,
    filter_out_uncommon_symbols,
    strip_text,
]
