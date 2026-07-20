//! ngram_tfidf_rs — character n-gram TF-IDF vectorizer, sklearn-parity.
//!
//! Replicates string_grouper's vectorization pipeline, in upstream order:
//!   1. lowercase (if `ignore_case`)                      [str.lower()]
//!   2. NFKD -> ASCII, dropping non-ASCII (if `normalize_ascii`)
//!      [unicodedata.normalize('NFKD', s).encode('ASCII', 'ignore')]
//!   3. removal of the characters in `removed_chars`
//!      (upstream default regex `[,-./]|\s` is a plain character class:
//!      pass ",-./" and whitespace is always removed; arbitrary regexes
//!      are NOT supported — documented limitation)
//!   4. character n-grams of fixed size
//!   5. TF-IDF with scikit-learn defaults:
//!        idf(t) = ln((1 + n_docs) / (1 + df(t))) + 1   (smooth_idf=True)
//!        tf = raw count (sublinear_tf=False), then row-wise L2 norm
//!
//! Outputs are numpy arrays (zero Python-list overhead): CSR triplets
//! (indptr, indices, data) + vocabulary (list of str) + idf vector.

use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use regex::Regex;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use unicode_normalization::UnicodeNormalization;

const DEFAULT_REGEX: &str = r"[,-./]|\s";

/// Cleanup strategy for step 3: fast character filter for the upstream
/// default pattern, full regex engine otherwise.
enum Cleaner {
    Default,
    Custom(Regex),
}

impl Cleaner {
    fn compile(pattern: &str) -> PyResult<Self> {
        if pattern == DEFAULT_REGEX {
            return Ok(Cleaner::Default);
        }
        Regex::new(pattern).map(Cleaner::Custom).map_err(|e| {
            PyValueError::new_err(format!(
                "regex not supported by the Rust engine (Python-specific \
syntax such as lookaround/backreference?): {e}"
            ))
        })
    }
}

#[inline]
fn preprocess(s: &str, ignore_case: bool, normalize_ascii: bool, cleaner: &Cleaner) -> String {
    // 1. lowercase
    let lowered: String = if ignore_case {
        s.chars().flat_map(|c| c.to_lowercase()).collect()
    } else {
        s.to_string()
    };
    // 2. NFKD -> ASCII ignore
    let folded: String = if normalize_ascii {
        lowered.nfkd().filter(|c| c.is_ascii()).collect()
    } else {
        lowered
    };
    // 3. removal of regex matches (upstream: re.sub(regex, '', s))
    match cleaner {
        Cleaner::Default => folded
            .chars()
            .filter(|c| !c.is_whitespace() && ![',', '-', '.', '/'].contains(c))
            .collect(),
        Cleaner::Custom(re) => re.replace_all(&folded, "").into_owned(),
    }
}

/// n-gram counts over &str windows of the preprocessed doc: zero allocation
/// per n-gram (borrows from `doc`), FxHash instead of SipHash.
fn count_ngrams(doc: &str, n: usize) -> FxHashMap<&str, u32> {
    let mut counts: FxHashMap<&str, u32> = FxHashMap::default();
    let bounds: Vec<usize> = doc.char_indices().map(|(i, _)| i).chain([doc.len()]).collect();
    if bounds.len() > n {
        for w in bounds.windows(n + 1) {
            *counts.entry(&doc[w[0]..w[n]]).or_insert(0) += 1;
        }
    }
    counts
}

fn l2_normalize(indptr: &[i64], data: &mut [f64]) {
    let ranges: Vec<(usize, usize)> = indptr
        .windows(2)
        .map(|w| (w[0] as usize, w[1] as usize))
        .collect();
    ranges.into_par_iter().for_each(|(lo, hi)| {
        // Safety: row ranges are disjoint
        let row = unsafe {
            std::slice::from_raw_parts_mut(data.as_ptr().add(lo) as *mut f64, hi - lo)
        };
        let norm: f64 = row.iter().map(|v| v * v).sum::<f64>().sqrt();
        if norm > 0.0 {
            row.iter_mut().for_each(|v| *v /= norm);
        }
    });
}

/// Assemble CSR triplets from per-row (indices, data), then L2-normalize.
fn assemble_csr(rows: Vec<(Vec<i64>, Vec<f64>)>) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let nnz: usize = rows.iter().map(|(i, _)| i.len()).sum();
    let mut indptr: Vec<i64> = Vec::with_capacity(rows.len() + 1);
    let mut indices: Vec<i64> = Vec::with_capacity(nnz);
    let mut data: Vec<f64> = Vec::with_capacity(nnz);
    indptr.push(0);
    for (i, d) in rows {
        indices.extend_from_slice(&i);
        data.extend_from_slice(&d);
        indptr.push(indices.len() as i64);
    }
    l2_normalize(&indptr, &mut data);
    (indptr, indices, data)
}

type FitOut<'py> = (
    &'py PyArray1<i64>,
    &'py PyArray1<i64>,
    &'py PyArray1<f64>,
    Vec<String>,
    &'py PyArray1<f64>,
);

#[pyfunction]
#[pyo3(signature = (docs, ngram_size, regex="[,-./]|\\s", ignore_case=true, normalize_ascii=true))]
fn fit_transform<'py>(
    py: Python<'py>,
    docs: Vec<String>,
    ngram_size: usize,
    regex: &str,
    ignore_case: bool,
    normalize_ascii: bool,
) -> PyResult<FitOut<'py>> {
    let cleaner = Cleaner::compile(regex)?;
    let (indptr, indices, data, vocab, idf) = py.allow_threads(|| {
        let n_docs = docs.len();

        let cleaned: Vec<String> = docs
            .par_iter()
            .map(|d| preprocess(d, ignore_case, normalize_ascii, &cleaner))
            .collect();
        let per_doc: Vec<FxHashMap<&str, u32>> = cleaned
            .par_iter()
            .map(|d| count_ngrams(d, ngram_size))
            .collect();

        // vocabulary + document frequencies (sklearn: lexicographic order)
        let mut df: FxHashMap<&str, u32> = FxHashMap::default();
        for counts in &per_doc {
            for key in counts.keys() {
                *df.entry(key).or_insert(0) += 1;
            }
        }
        let mut vocab_sorted: Vec<&str> = df.keys().copied().collect();
        vocab_sorted.sort_unstable();
        let vocab_index: FxHashMap<&str, usize> = vocab_sorted
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i))
            .collect();

        let idf: Vec<f64> = vocab_sorted
            .iter()
            .map(|t| ((1.0 + n_docs as f64) / (1.0 + df[t] as f64)).ln() + 1.0)
            .collect();

        let rows: Vec<(Vec<i64>, Vec<f64>)> = per_doc
            .par_iter()
            .map(|counts| {
                let mut cols: Vec<(usize, u32)> = counts
                    .iter()
                    .map(|(t, c)| (vocab_index[*t], *c))
                    .collect();
                cols.sort_unstable_by_key(|&(j, _)| j);
                let indices: Vec<i64> = cols.iter().map(|&(j, _)| j as i64).collect();
                let data: Vec<f64> = cols.iter().map(|&(j, c)| c as f64 * idf[j]).collect();
                (indices, data)
            })
            .collect();

        let (indptr, indices, data) = assemble_csr(rows);
        let vocab_out: Vec<String> = vocab_sorted.iter().map(|s| s.to_string()).collect();
        (indptr, indices, data, vocab_out, idf)
    });

    Ok((
        indptr.into_pyarray(py),
        indices.into_pyarray(py),
        data.into_pyarray(py),
        vocab,
        idf.into_pyarray(py),
    ))
}

type TransformOut<'py> = (&'py PyArray1<i64>, &'py PyArray1<i64>, &'py PyArray1<f64>);

#[pyfunction]
#[pyo3(signature = (docs, vocab, idf, ngram_size, regex="[,-./]|\\s", ignore_case=true, normalize_ascii=true))]
fn transform<'py>(
    py: Python<'py>,
    docs: Vec<String>,
    vocab: Vec<String>,
    idf: Vec<f64>,
    ngram_size: usize,
    regex: &str,
    ignore_case: bool,
    normalize_ascii: bool,
) -> PyResult<TransformOut<'py>> {
    let cleaner = Cleaner::compile(regex)?;
    let (indptr, indices, data) = py.allow_threads(|| {
        let vocab_index: FxHashMap<&str, usize> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i))
            .collect();

        let cleaned: Vec<String> = docs
            .par_iter()
            .map(|d| preprocess(d, ignore_case, normalize_ascii, &cleaner))
            .collect();
        let rows: Vec<(Vec<i64>, Vec<f64>)> = cleaned
            .par_iter()
            .map(|d| {
                let counts = count_ngrams(d, ngram_size);
                let mut cols: Vec<(usize, u32)> = counts
                    .iter()
                    .filter_map(|(t, c)| vocab_index.get(*t).map(|&j| (j, *c)))
                    .collect();
                cols.sort_unstable_by_key(|&(j, _)| j);
                let indices: Vec<i64> = cols.iter().map(|&(j, _)| j as i64).collect();
                let data: Vec<f64> = cols.iter().map(|&(j, c)| c as f64 * idf[j]).collect();
                (indices, data)
            })
            .collect();
        assemble_csr(rows)
    });
    Ok((
        indptr.into_pyarray(py),
        indices.into_pyarray(py),
        data.into_pyarray(py),
    ))
}

#[pymodule]
fn _ngram_tfidf(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_transform, m)?)?;
    m.add_function(wrap_pyfunction!(transform, m)?)?;
    Ok(())
}
