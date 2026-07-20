# Benchmarks

Reproduction of the performance figures reported in the main README.

## Dataset

The reference benchmark uses the SEC EDGAR company names dataset
(~663k names), the same one used by string_grouper's documentation.
Download `sec_edgar_company_info.csv` (a public dataset, widely mirrored;
e.g. on Kaggle under "SEC EDGAR Companies List") and set its path in the
first cell of the notebook (`COMPANY_NAMES`).

## Running

```bash
pip install 'string-grouperx[test]'          # sgx + polars/pandas/pyarrow/sklearn
pip install "git+https://github.com/Bergvca/string_grouper.git@add_sp_matmul_rs"
jupyter lab bench_string_grouperx.ipynb
```

The notebook runs 4 configurations (best-of-N on wall time) and prints:
- a styled comparison table (wall time, total CPU, parallelism, speedup),
- a decomposition of the gain at equal matmul backend,
- a parity check against string_grouper.

`time.process_time()` is used for CPU accounting so the parallelism column
(CPU/wall) correctly reflects the Rust/rayon threads.

## Note

Figures depend on core count: the Rust vectorizer gain comes from filling a
single-core gap in string_grouper's Python vectorization, so the speedup
grows with the number of cores and shrinks toward 1x on a single core.
Report your machine's core count alongside any figures.
