# string-grouperx

Fuzzy string matching and grouping (TF-IDF over character n-grams + sparse
top-n cosine similarity), **dataframe-agnostic** and **pandas-free**: a rewrite
of [string_grouper](https://github.com/Bergvca/string_grouper)'s functional API
with a Rust vectorization core.

> **Origin.** `string-grouperx` is a derivative rewrite of
> [string_grouper](https://github.com/Bergvca/string_grouper)
> (Chris van den Berg, MIT license). It preserves the functional API and the
> results (parity verified, see below), while replacing the pandas +
> scikit-learn core with a Rust + narwhals pipeline. The similarity computation
> backend relies on [`sp_matmul_rs`](https://pypi.org/project/sp-matmul-rs/)
> (cf. string_grouper PR #105).

```python
import polars as pl
from string_grouperx import match_strings, group_similar_strings

s = pl.Series('company_name', ['Hôpital de Brest', 'Hopital de Brest',
                               'CHU Brest', 'Clinique Kerlédé'])

match_strings(s, min_similarity=0.6)          # -> pl.DataFrame
group_similar_strings(s, min_similarity=0.6)  # -> pl.DataFrame (group representatives)
```

Output follows input: pass a `pl.Series` -> Polars output; a `pd.Series` ->
pandas output; a `pyarrow.ChunkedArray` -> PyArrow output, any eager backend
supported by [narwhals](https://narwhals-dev.github.io/narwhals/)
(cuDF, Modin, ...) works identically.

## Architecture

| Layer | Role | Technology |
|---|---|---|
| Boundary | ingestion / native output construction | narwhals |
| Vectorization | character n-grams + TF-IDF (sklearn formula) | Rust (pyo3 + rayon, embedded) |
| Similarity | sparse top-n cosine | sp_matmul_rs |
| Post-processing | connected components, representatives, best match | numpy / scipy |

No pandas dependency. The Rust core handles custom cleanup regexes natively
(crate `regex`, cost ~identical to the default: ~195-225 ms vs 193 ms on 40k
strings).

## API

The four string_grouper functions, same signatures, same output column names:

- `match_strings(master, duplicates=None, master_id=None, duplicates_id=None, **options)`
- `match_most_similar(master, duplicates, master_id=None, duplicates_id=None, **options)`
- `group_similar_strings(strings_to_group, string_ids=None, **options)`
- `compute_pairwise_similarities(s1, s2, **options)`

Options (see `Config`): `ngram_size`, `regex`, `ignore_case`,
`normalize_to_ascii`, `min_similarity`, `max_n_matches`, `group_rep`
(`'centroid'` default / `'first'`), `ignore_index`, `force_symmetries`,
`number_of_processes`, `chunk_cols`, `vectorizer` (`'auto'` default /
`'rust'` / `'sklearn'`).

## Fidelity to string_grouper

The pipeline replicates string_grouper 0.7.x (branch `add_sp_matmul_rs`):
n-gram analyzer (lowercase, NFKD->ASCII, cleanup), TfidfVectorizer (smooth idf,
L2 norm), self-join symmetrization, representative schemes and tie-breaking rules.

**Parity guarantee** (validated on SEC EDGAR, 663k company names, the reference
benchmark of string_grouper): with a non-saturated `max_n_matches`, the same set
of pairs (identical indices and strings) and similarities equal up to
floating-point summation noise (max observed 1.2e-15). 

Not ported (intentional scope): `include_zeroes` (only relevant when
`min_similarity <= 0`), `replace_na`, the `sparse_dot_topn`/`n_blocks` backend,
the `StringGrouper` class (add_match/remove_match/cache).

## Performance

SEC EDGAR, ~663k company names, self-join, `min_similarity=0.8`, backend
`sp_matmul_rs` throughout (measurements on an 8-core machine):

| | wall time | notes |
|---|---|---|
| string_grouper (sparse_dot_topn) | ~67 s | historical baseline |
| string_grouper (sp_matmul_rs, PR #105) | ~37 s | matmul backend |
| string-grouperx, sklearn vectorizer | ~34 s | pandas-free rewrite gain |
| string-grouperx, rust vectorizer | **~27 s** | full config |

At equal matmul backend, the gain decomposes into two independent effects: the
pandas-free rewrite (native numpy/narwhals outputs instead of pandas
groupby/merge post-processing) accounts for ~10%, and the Rust vectorizer for
~19% more. Crucially, total CPU time stays roughly constant across
configurations (~3min30): the wall-clock gain comes from **parallelism**, not
from doing less work. string_grouper's TF-IDF analyzer is pure Python under the
GIL and runs single-threaded, leaving cores idle during vectorization;
string-grouperx moves vectorization to Rust/rayon with the GIL released, filling
that gap. The speedup therefore **grows with core count** and shrinks toward 1x
on a single core.

Benchmark notebook: see `bench/`.

## Installation / build

```bash
pip install maturin
maturin build --release
pip install target/wheels/*.whl

# extras
pip install 'string-grouperx[sklearn]'   # fallback backend for custom regexes
pip install 'string-grouperx[test]' && pytest tests/
```

Toolchain note: `rayon` is pinned to 1.8.1 to build with rustc as old as 1.75
(Ubuntu 24); this pin can be lifted with a recent toolchain.

