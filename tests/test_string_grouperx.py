"""Tests string_grouperx.

Trois niveaux :
1. comportement de l'API sur petit corpus (autonome, aucun extra requis
   au-delà des dépendances de test pandas/polars/pyarrow)
2. accord des backends vectorizer rust vs sklearn (skip si sklearn absent)
3. parité stricte vs string_grouper upstream (skip si non installé)
"""
import numpy as np
import pytest

pd = pytest.importorskip('pandas')
pl = pytest.importorskip('polars')

from string_grouperx import (
    Config,
    compute_pairwise_similarities,
    group_similar_strings,
    match_most_similar,
    match_strings,
)

COMPANIES = [
    'Hôpital de Brest', 'Hopital de Brest', 'CHU Brest', 'C.H.U. de Brest',
    'Clinique Kerlédé', 'Clinique Kerlede', 'CHRU Brest',
    'Centre Hospitalier de Quimper', 'CH Quimper', 'CH de Quimper',
    'Polyclinique de Quimper', 'Hopital de Morlaix', 'CH Morlaix',
]
IDS = [f'F{i:03d}' for i in range(len(COMPANIES))]
DUPES = ['Hopital Brest', 'CH Kemper', 'inconnu']


@pytest.fixture
def pl_s():
    return pl.Series('name', COMPANIES)


@pytest.fixture
def pd_s():
    return pd.Series(COMPANIES, name='name')


# ---------------------------------------------------------------- niveau 1

def test_match_strings_polars_roundtrip(pl_s):
    out = match_strings(pl_s, min_similarity=0.6)
    assert isinstance(out, pl.DataFrame)
    assert out.columns == ['left_index', 'left_name', 'similarity',
                           'right_name', 'right_index']
    # diagonale présente (self-match à ~1.0) et symétrie
    diag = out.filter(pl.col('left_index') == pl.col('right_index'))
    assert diag.height == len(COMPANIES)
    pairs = set(zip(out['left_index'], out['right_index']))
    assert all((r, l) in pairs for l, r in pairs)
    # les accents sont normalisés : Hôpital ~ Hopital au-dessus du seuil
    assert out.filter((pl.col('left_index') == 0) & (pl.col('right_index') == 1)).height == 1


def test_output_follows_input(pl_s, pd_s):
    assert isinstance(match_strings(pl_s, min_similarity=0.6), pl.DataFrame)
    assert isinstance(match_strings(pd_s, min_similarity=0.6), pd.DataFrame)
    pa = pytest.importorskip('pyarrow')
    out = match_strings(pa.chunked_array([COMPANIES]), min_similarity=0.6)
    assert isinstance(out, pa.Table)


def test_match_strings_with_ids(pl_s):
    out = match_strings(pl_s, master_id=pl.Series('id', IDS), min_similarity=0.6)
    assert out.columns == ['left_index', 'left_name', 'left_id', 'similarity',
                           'right_id', 'right_name', 'right_index']
    row = out.filter((pl.col('left_index') == 0) & (pl.col('right_index') == 1))
    assert row['left_id'].item() == 'F000' and row['right_id'].item() == 'F001'


def test_group_similar_strings(pl_s):
    out = group_similar_strings(pl_s, min_similarity=0.6)
    assert out.columns == ['group_rep_index', 'group_rep_name']
    assert out.height == len(COMPANIES)
    # les membres d'un même groupe partagent le même représentant
    assert out['group_rep_index'][0] == out['group_rep_index'][1]
    # group_rep='first' -> représentant = plus petit index du groupe
    first = group_similar_strings(pl_s, min_similarity=0.6, group_rep='first')
    assert first['group_rep_index'][1] == 0


def test_group_ignore_index_returns_series(pl_s):
    out = group_similar_strings(pl_s, min_similarity=0.6, ignore_index=True)
    assert isinstance(out, pl.Series)
    assert len(out) == len(COMPANIES)


def test_match_most_similar(pl_s):
    out = match_most_similar(pl_s, pl.Series('name', DUPES), min_similarity=0.5)
    got = out['most_similar_name'].to_list()
    assert got[0] in COMPANIES          # 'Hopital Brest' trouve un master
    assert got[2] == 'inconnu'          # pas de match -> le doublon lui-même
    assert out['most_similar_index'][2] is None or np.isnan(out['most_similar_index'][2])


def test_compute_pairwise(pl_s):
    other = pl.Series('o', ['Hopital Brest'] * len(COMPANIES))
    out = compute_pairwise_similarities(pl_s, other)
    assert isinstance(out, pl.Series) and len(out) == len(COMPANIES)
    assert 0.0 <= out.min() and out.max() <= 1.0 + 1e-12
    with pytest.raises(Exception):
        compute_pairwise_similarities(pl_s, pl.Series('o', ['x']))


def test_edge_cases_strings():
    s = pl.Series('name', ['Hôpital de Brest', '', 'ab', ',-./  ', 'CHU'])
    out = match_strings(s, min_similarity=0.3)
    assert isinstance(out, pl.DataFrame)  # ne lève pas, lignes vides tolérées


def test_custom_regex_rust_native(pl_s):
    # le moteur Rust gère les regex custom sans sklearn
    out = match_strings(pl_s, min_similarity=0.6, regex=r'[,-./&]|\s',
                        vectorizer='rust')
    assert isinstance(out, pl.DataFrame)


@pytest.mark.parametrize('rx', [r'[,-./]|\s', r'[^a-z0-9]', r'\d|\s',
                                r"[,-./']|\s", r'[\s,;:]'])
def test_custom_regex_parity_rust_vs_sklearn(pl_s, rx):
    pytest.importorskip('sklearn')
    rust = match_strings(pl_s, min_similarity=0.4, regex=rx, vectorizer='rust')
    sk = match_strings(pl_s, min_similarity=0.4, regex=rx, vectorizer='sklearn')
    assert rust['left_index'].to_list() == sk['left_index'].to_list()
    assert rust['right_index'].to_list() == sk['right_index'].to_list()
    assert np.allclose(rust['similarity'], sk['similarity'], atol=1e-12)


def test_python_specific_regex_raises_or_falls_back(pl_s):
    lookbehind = r'(?<=A),'   # non supporté par le moteur regex Rust
    with pytest.raises(ValueError, match='regex'):
        match_strings(pl_s, min_similarity=0.6, regex=lookbehind, vectorizer='rust')
    pytest.importorskip('sklearn')
    out = match_strings(pl_s, min_similarity=0.6, regex=lookbehind)  # auto -> repli
    assert isinstance(out, pl.DataFrame)


# ---------------------------------------------------------------- niveau 2

def test_rust_and_sklearn_backends_agree(pl_s):
    pytest.importorskip('sklearn')
    for fn, kw in [
        (match_strings, {}),
        (group_similar_strings, {}),
        (group_similar_strings, {'group_rep': 'first'}),
    ]:
        rust = fn(pl_s, min_similarity=0.5, vectorizer='rust', **kw)
        sk = fn(pl_s, min_similarity=0.5, vectorizer='sklearn', **kw)
        assert rust.to_pandas().equals(sk.to_pandas()), fn.__name__


# ---------------------------------------------------------------- niveau 3

def test_parity_with_upstream_string_grouper():
    """Parité vs upstream, méthodologie validée sur SEC EDGAR (663k chaînes).

    Garantie : à `max_n_matches` non saturé, même ensemble de paires
    (index et chaînes exacts) et similarités égales à la tolérance
    flottante près (~1e-15 observé ; deux implémentations ne somment pas
    dans le même ordre, l'addition flottante n'est pas associative).
    Avec coupure saturée, seul le départage des ex-aequo à l'ULP peut
    différer — même cardinal, paires équivalentes.
    L'égalité bitwise (`equals`) n'est PAS la bonne comparaison
    inter-implémentations.
    """
    up = pytest.importorskip('string_grouper')
    pd_s = pd.Series(COMPANIES, name='name')
    pl_s = pl.Series('name', COMPANIES)
    key = ['left_index', 'right_index']

    # match_strings : ensembles de paires + similarité à tolérance
    ref = (up.match_strings(pd_s, min_similarity=0.6, max_n_matches=10_000)
             .reset_index(drop=True).sort_values(key).reset_index(drop=True))
    out = (match_strings(pl_s, min_similarity=0.6, max_n_matches=10_000)
             .to_pandas().sort_values(key).reset_index(drop=True))
    assert ref[key + ['left_name', 'right_name']].equals(
        out[key + ['left_name', 'right_name']])
    assert np.allclose(ref['similarity'], out['similarity'], atol=1e-12, rtol=0)

    # group / match_most_similar : sorties déterministes hors similarité
    ref_g = up.group_similar_strings(pd_s, min_similarity=0.6).reset_index(drop=True)
    out_g = group_similar_strings(pl_s, min_similarity=0.6).to_pandas()
    assert ref_g.equals(out_g)

    ref_m = up.match_most_similar(pd_s, pd.Series(DUPES, name='name'),
                                  min_similarity=0.5).reset_index(drop=True)
    out_m = match_most_similar(pl_s, pl.Series('name', DUPES),
                               min_similarity=0.5).to_pandas()
    assert ref_m.equals(out_m)
