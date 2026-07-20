"""string_grouperx — fuzzy string matching and grouping, dataframe-agnostic.

Pandas-free rewrite of string_grouper's functional API:
TF-IDF character n-grams (Rust core) + sparse cosine top-n (sp_matmul_rs),
accepting and returning any eager dataframe backend supported by narwhals
(Polars, pandas, PyArrow, cuDF, Modin, ...) — output follows input.

>>> import polars as pl
>>> from string_grouperx import match_strings
>>> s = pl.Series('name', ['Hôpital de Brest', 'Hopital de Brest', 'CHU Brest'])
>>> match_strings(s, min_similarity=0.6)   # -> pl.DataFrame
"""

from string_grouperx._core import (
    Config,
    compute_pairwise_similarities,
    group_similar_strings,
    match_most_similar,
    match_strings,
)

__version__ = "0.2.0"

__all__ = [
    "Config",
    "match_strings",
    "match_most_similar",
    "group_similar_strings",
    "compute_pairwise_similarities",
    "__version__",
]
