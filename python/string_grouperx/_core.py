"""string_grouperx — Vx: pandas-free rewrite of string_grouper's functional API.

Architecture
------------
* ``_core``-level functions: pure numpy / scipy / scikit-learn / sp_matmul_rs.
  No dataframe library anywhere. Inputs are lists of strings, outputs are
  numpy arrays (match triplets, group representatives, ...).
* Boundary: narwhals. Inputs may come from ANY eager backend narwhals
  supports (Polars, pandas, PyArrow, cuDF, Modin, ...); outputs are
  constructed *natively* in the input backend via ``nw.from_dict`` /
  ``nw.new_series``. There is no pandas conversion at any point — pandas is
  just one more backend, not a dependency.

Faithfulness
------------
The numerical pipeline replicates string_grouper 0.7.x (branch
``add_sp_matmul_rs``) exactly: same n-gram analyzer (lowercase, NFKD->ASCII,
regex cleanup), same TfidfVectorizer setup, same ``sp_matmul_topn_rs`` call,
same diagonal-fix + symmetrisation for self-joins, same group-representative
schemes ('centroid' default, 'first'), same tie-breaking rules, same output
column names (``left_index``, ``left_<name>``, ``similarity``, ...,
``group_rep_index``, ``most_similar_<name>``, ...).

Scope of the prototype (documented, not hidden)
-----------------------------------------------
Supported: the 4 public functions with ids, plus options ngram_size, regex,
ignore_case, normalize_to_ascii, min_similarity, max_n_matches, group_rep,
ignore_index, force_symmetries, number_of_processes, chunk_cols.
Not (yet) ported: include_zeroes (only relevant when min_similarity <= 0),
replace_na, the sparse_dot_topn/n_blocks backend, the StringGrouper class
API (add_match / remove_match / caching).
"""

import re
import multiprocessing
from dataclasses import dataclass, replace as dc_replace
from typing import Optional, Tuple
from unicodedata import normalize as _unicode_normalize

import numpy as np
import narwhals as nw
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sp_matmul_rs import sp_matmul_topn as sp_matmul_topn_rs

DEFAULT_COLUMN_NAME = 'side'
DEFAULT_ID_NAME = 'id'
DEFAULT_MASTER_NAME = 'master'
LEFT, RIGHT = 'left_', 'right_'
GROUP_REP_PREFIX = 'group_rep_'
MOST_SIMILAR_PREFIX = 'most_similar_'


@dataclass(frozen=True)
class Config:
    ngram_size: int = 3
    regex: str = r'[,-./]|\s'
    ignore_case: bool = True
    normalize_to_ascii: bool = True
    min_similarity: float = 0.8
    max_n_matches: int = 20
    group_rep: str = 'centroid'
    ignore_index: bool = False
    force_symmetries: bool = True
    number_of_processes: int = max(multiprocessing.cpu_count() - 1, 1)
    chunk_cols: Optional[int] = None
    vectorizer: str = 'auto'  # 'auto' | 'rust' | 'sklearn' ('auto' = rust si dispo)


# --------------------------------------------------------------------------
# narwhals boundary
# --------------------------------------------------------------------------

def _ingest(obj, arg: str) -> Tuple[list, Optional[str], nw.Implementation]:
    """Any supported native Series -> (python list, name-or-None, backend)."""
    series = nw.from_native(obj, series_only=True)
    name = series.name if series.name else None  # '' (polars) and None (pandas) -> None
    return series.to_list(), name, series.implementation


def _emit_frame(columns: dict, backend: nw.Implementation):
    return nw.from_dict(columns, backend=backend).to_native()


def _emit_series(name: str, values, backend: nw.Implementation):
    return nw.new_series(name, values, backend=backend).to_native()


# --------------------------------------------------------------------------
# core: numpy / scipy / sklearn / rust only
# --------------------------------------------------------------------------

def _make_analyzer(cfg: Config):
    def n_grams(string: str):
        if cfg.ignore_case and string is not None:
            string = string.lower()
        if cfg.normalize_to_ascii:
            string = _unicode_normalize('NFKD', string).encode('ASCII', 'ignore').decode()
        string = re.sub(cfg.regex, '', string)
        grams = zip(*[string[i:] for i in range(cfg.ngram_size)])
        return [''.join(g) for g in grams]
    return n_grams


from string_grouperx import _ngram_tfidf as _rs_vectorizer


def _tfidf_matrices(master: list, duplicates: Optional[list], cfg: Config):
    if cfg.vectorizer == 'sklearn':
        return _tfidf_matrices_sklearn(master, duplicates, cfg)
    try:
        return _tfidf_matrices_rust(master, duplicates, cfg)
    except ValueError as exc:
        # The Rust regex engine rejects Python-specific syntax
        # (lookaround, backreferences). In 'auto' mode, fall back to sklearn.
        if 'regex' in str(exc) and cfg.vectorizer == 'auto':
            return _tfidf_matrices_sklearn(master, duplicates, cfg)
        raise


def _tfidf_matrices_sklearn(master: list, duplicates: Optional[list], cfg: Config):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError as exc:
        raise ImportError(
            "Le backend sklearn requiert scikit-learn : "
            "pip install string-grouperx[sklearn]") from exc
    vectorizer = TfidfVectorizer(min_df=1, analyzer=_make_analyzer(cfg))
    corpus = master if duplicates is None else master + duplicates
    vectorizer.fit(corpus)
    master_matrix = vectorizer.transform(master)
    duplicate_matrix = vectorizer.transform(duplicates) if duplicates is not None else master_matrix
    return master_matrix, duplicate_matrix


def _tfidf_matrices_rust(master: list, duplicates: Optional[list], cfg: Config):
    """Embedded Rust backend: exact numerical parity with sklearn."""
    corpus = master if duplicates is None else master + duplicates
    indptr, indices, data, vocab, _idf = _rs_vectorizer.fit_transform(
        corpus, cfg.ngram_size,
        regex=cfg.regex,
        ignore_case=cfg.ignore_case,
        normalize_ascii=cfg.normalize_to_ascii,
    )
    matrix = csr_matrix((data, indices, indptr), shape=(len(corpus), len(vocab)))
    if duplicates is None:
        return matrix, matrix
    return matrix[:len(master)], matrix[len(master):]


def _match_arrays(master: list, duplicates: Optional[list], cfg: Config):
    """Return (rows, cols, sims): master-index, dupe-index, cosine similarity."""
    master_matrix, duplicate_matrix = _tfidf_matrices(master, duplicates, cfg)
    matches = sp_matmul_topn_rs(
        master_matrix,
        duplicate_matrix.transpose(),
        top_n=cfg.max_n_matches,
        threshold=cfg.min_similarity,
        sort=True,
        n_threads=cfg.number_of_processes,
        chunk_cols=cfg.chunk_cols,
    )
    if duplicates is None and cfg.force_symmetries:
        matches = matches.tolil()
        diag = np.arange(matches.shape[0])
        matches[diag, diag] = 1                       # _fix_diagonal
        r, c = matches.nonzero()
        matches[c, r] = matches[r, c]                 # _symmetrize_matrix
        matches = matches.tocsr()
    coo = matches.tocoo()
    return (coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data.astype(np.float64))


def _group_representatives(n: int, rows, cols, sims, cfg: Config) -> np.ndarray:
    """Replicates _deduplicate: connected components + rep per group.

    'first'    -> lowest element index in the group
    'centroid' -> element with highest similarity row-sum (ties: lowest index)
    Returns rep[i] = index of the representative of i's group.
    """
    graph = csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    _, groups = connected_components(csgraph=graph, directed=True)

    indices = np.arange(n)
    if cfg.group_rep == 'centroid':
        sim_graph = csr_matrix((sims, (rows, cols)), shape=(n, n))
        weight = np.asarray(sim_graph.sum(axis=1)).squeeze(axis=1)
        order = np.lexsort((indices, -weight, groups))  # by group, weight desc, index asc
    elif cfg.group_rep == 'first':
        order = np.argsort(groups, kind='stable')       # by group, index asc
    else:
        raise ValueError(f"group_rep must be 'centroid' or 'first', got {cfg.group_rep!r}")

    sorted_groups = groups[order]
    block_start = np.r_[True, sorted_groups[1:] != sorted_groups[:-1]]
    rep_of_block = order[block_start]                   # winner of each group block
    rep_sorted = rep_of_block[np.cumsum(block_start) - 1]
    rep = np.empty(n, dtype=np.int64)
    rep[order] = rep_sorted
    return rep


def _best_master_per_dupe(n_dupes: int, rows, cols, sims):
    """Replicates _get_nearest_matches' selection: per duplicate, the master
    with max similarity; ties broken by lowest master index.
    Returns (dupe_indices_with_match, master_index_for_them)."""
    if len(cols) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    order = np.lexsort((rows, -sims, cols))             # by dupe, sim desc, master asc
    sorted_cols = cols[order]
    first = np.r_[True, sorted_cols[1:] != sorted_cols[:-1]]
    return sorted_cols[first], rows[order][first]


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------

def match_strings(master, duplicates=None, master_id=None, duplicates_id=None, **kwargs):
    cfg = Config(**kwargs)
    m_values, m_name, backend = _ingest(master, 'master')
    self_join = duplicates is None
    if self_join:
        d_values, d_name = m_values, m_name
    else:
        d_values, d_name, _ = _ingest(duplicates, 'duplicates')

    rows, cols, sims = _match_arrays(m_values, None if self_join else d_values, cfg)

    left_name = m_name or DEFAULT_COLUMN_NAME
    right_name = d_name or DEFAULT_COLUMN_NAME
    m_arr = np.asarray(m_values, dtype=object)
    d_arr = np.asarray(d_values, dtype=object)

    columns = {}
    if not cfg.ignore_index:
        columns[f'{LEFT}index'] = rows
    columns[f'{LEFT}{left_name}'] = m_arr[rows].tolist()
    if master_id is not None:
        mid_values, mid_name, _ = _ingest(master_id, 'master_id')
        did_values, did_name = (mid_values, mid_name) if self_join \
            else _ingest(duplicates_id, 'duplicates_id')[:2]
        columns[f'{LEFT}{mid_name or DEFAULT_ID_NAME}'] = \
            np.asarray(mid_values, dtype=object)[rows].tolist()
        columns['similarity'] = sims
        columns[f'{RIGHT}{did_name or DEFAULT_ID_NAME}'] = \
            np.asarray(did_values, dtype=object)[cols].tolist()
    else:
        columns['similarity'] = sims
    columns[f'{RIGHT}{right_name}'] = d_arr[cols].tolist()
    if not cfg.ignore_index:
        columns[f'{RIGHT}index'] = cols
    return _emit_frame(columns, backend)


def group_similar_strings(strings_to_group, string_ids=None, **kwargs):
    cfg = Config(**kwargs)
    values, name, backend = _ingest(strings_to_group, 'strings_to_group')
    rows, cols, sims = _match_arrays(values, None, cfg)
    rep = _group_representatives(len(values), rows, cols, sims, cfg)

    label = f'{GROUP_REP_PREFIX}{name}' if name else GROUP_REP_PREFIX[:-1]
    arr = np.asarray(values, dtype=object)
    columns = {}
    if string_ids is not None:
        id_values, id_name, _ = _ingest(string_ids, 'string_ids')
        columns[f'{GROUP_REP_PREFIX}{id_name or DEFAULT_ID_NAME}'] = \
            np.asarray(id_values, dtype=object)[rep].tolist()
    if not cfg.ignore_index:
        columns[f'{GROUP_REP_PREFIX}index'] = rep
    columns[label] = arr[rep].tolist()

    if len(columns) == 1:  # ignore_index=True, no ids -> Series (upstream squeeze)
        return _emit_series(label, columns[label], backend)
    return _emit_frame(columns, backend)


def match_most_similar(master, duplicates, master_id=None, duplicates_id=None, **kwargs):
    kwargs['max_n_matches'] = 1
    cfg = Config(**kwargs)
    m_values, m_name, backend = _ingest(master, 'master')
    d_values, _, _ = _ingest(duplicates, 'duplicates')
    rows, cols, sims = _match_arrays(m_values, d_values, cfg)
    matched_dupes, matched_masters = _best_master_per_dupe(len(d_values), rows, cols, sims)

    n = len(d_values)
    master_label = f'{MOST_SIMILAR_PREFIX}{m_name or DEFAULT_MASTER_NAME}'
    out_strings = np.asarray(d_values, dtype=object).copy()          # fallback: the dupe itself
    out_strings[matched_dupes] = np.asarray(m_values, dtype=object)[matched_masters]

    columns = {}
    if not cfg.ignore_index:
        out_index = np.full(n, np.nan)                               # NaN when no match (upstream)
        out_index[matched_dupes] = matched_masters.astype(np.float64)
        columns[f'{MOST_SIMILAR_PREFIX}index'] = out_index
    if master_id is not None:
        mid_values, mid_name, _ = _ingest(master_id, 'master_id')
        did_values, _, _ = _ingest(duplicates_id, 'duplicates_id')
        out_ids = np.asarray(did_values, dtype=object).copy()        # fallback: duplicates_id
        out_ids[matched_dupes] = np.asarray(mid_values, dtype=object)[matched_masters]
        id_label = f'{MOST_SIMILAR_PREFIX}{mid_name or f"{DEFAULT_MASTER_NAME}_{DEFAULT_ID_NAME}"}'
        columns[id_label] = out_ids.tolist()
    columns[master_label] = out_strings.tolist()

    if len(columns) == 1:
        return _emit_series(master_label, columns[master_label], backend)
    return _emit_frame(columns, backend)


def compute_pairwise_similarities(string_series_1, string_series_2, **kwargs):
    cfg = Config(**kwargs)
    values_1, _, backend = _ingest(string_series_1, 'string_series_1')
    values_2, _, _ = _ingest(string_series_2, 'string_series_2')
    if len(values_1) != len(values_2):
        raise Exception('To perform this function, both input Series must have the same length.')
    m, d = _tfidf_matrices(values_1, values_2, cfg)
    sims = np.asarray(m.multiply(d).sum(axis=1)).squeeze(axis=1)
    return _emit_series('similarity', sims, backend)
