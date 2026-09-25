"""Pytest suite for Project Lazarus core, ML features, and RL env."""
from __future__ import annotations

import random

import numpy as np
import pytest

from lazarus.core import adna, crispr, phylogeny
from lazarus.data.species_db import GENE_TEMPLATES, SPECIES, candidate_score, ranked
from lazarus.data.synthetic import Read, random_sequence, simulate_read_set
from lazarus.ml.features import make_dataset, profile_matrix, read_feature_vector
from lazarus.rl.genome_env import OBS_DIM, GenomeGapFillEnv, RandomFiller, run_episode


# --------------------------------------------------------------------------- synthetic

def test_simulate_read_set_shapes_and_damage():
    rs = simulate_read_set(n_reads=300, damage_5p=0.35, damage_3p=0.3,
                           contamination=0.2, seed=1)
    assert rs.n == 300
    lengths = rs.lengths()
    assert lengths.min() >= 18
    anc = [r for r in rs.reads if r.is_ancient]
    assert 0.5 < len(anc) / rs.n < 1.0
    # heavy simulated damage must leave a terminal C→T footprint
    prof = adna.misincorporation_profile(anc)
    assert prof.ct_5p[0] > 0.15
    assert prof.ct_5p[0] > prof.ct_5p[-1]


def test_parse_reads_formats():
    fq = "@r1\nACGTACGTACGTACGTACGT\n+\nIIIIIIIIIIIIIIIIIIII\n"
    fa = ">a\nACGTACGTACGTACGTACGT\n>b\nACGTACGTACGTACGTACGT\n"
    bare = "ACGTACGTACGTACGTACGT\nACGTACGTACGTACGTACGT\n"
    assert len(adna.parse_reads(fq)) == 1
    assert len(adna.parse_reads(fa)) == 2
    assert len(adna.parse_reads(bare)) == 2


def test_analyze_bundle_keys():
    rs = simulate_read_set(n_reads=250, seed=3)
    out = adna.analyze(rs.reads)
    for key in ("profile", "length_stats", "damage_params", "authenticity_score", "verdict"):
        assert key in out
    assert 0.0 <= out["authenticity_score"] <= 100.0


# --------------------------------------------------------------------------- phylogeny

def test_kmer_distance_identical_zero():
    s = random_sequence(400, 0.4, random.Random(0))
    assert phylogeny.mash_distance(phylogeny.kmer_set(s), phylogeny.kmer_set(s)) == pytest.approx(0.0)


def test_nj_tree_contains_all_labels():
    labels = ["a", "b", "c", "d"]
    D = np.array([
        [0.0, 0.1, 0.3, 0.3],
        [0.1, 0.0, 0.3, 0.3],
        [0.3, 0.3, 0.0, 0.2],
        [0.3, 0.3, 0.2, 0.0],
    ])
    root = phylogeny.neighbor_joining(D, labels)
    nwk = phylogeny.to_newick(root)
    for lab in labels:
        assert lab in nwk
    lines, markers = phylogeny.tree_segments(root)
    assert lines and markers


def test_mds_shape():
    D = np.random.default_rng(0).random((5, 5))
    D = (D + D.T) / 2
    np.fill_diagonal(D, 0)
    coords = phylogeny.classical_mds(D, 2)
    assert coords.shape == (5, 2)


# --------------------------------------------------------------------------- crispr

def test_translate_roundtrip():
    dna = "ATGGCCTGTTAA"
    assert crispr.translate(dna) == "MAC*"


def test_plan_edits_on_template():
    tpl = GENE_TEMPLATES["mammoth_hbb"]
    plan = crispr.plan_edits(tpl.donor_dna, tpl.target_dna, gene_name=tpl.gene_name)
    assert plan.n_edits == 3
    assert all(s.guides or s.prime for s in plan.sites)
    assert crispr.translate(plan.target_dna) == plan.target_aa
    rows = plan.rows()
    assert len(rows) == 3 and "top guide" in rows[0]


def test_guide_designers():
    seq = GENE_TEMPLATES["dodo_mc1r"].donor_dna
    guides = crispr.design_guides(seq, edit_center=90, max_cut_distance=60)
    assert guides  # long gene must yield some SpCas9 sites
    best = guides[0]
    assert len(best.spacer) == 20 and len(best.pam) == 3
    assert best.pam.endswith("GG")
    peg = crispr.design_peg_rna(seq, crispr.extract_codon_edits(seq, GENE_TEMPLATES["dodo_mc1r"].target_dna)[0])
    assert "PBS (13 nt)" in peg


# --------------------------------------------------------------------------- ml features

def test_read_features_dim_and_range():
    r = Read(seq="ACGT" * 8, source="ACGT" * 8, pos=0, is_ancient=True)
    f = read_feature_vector(r)
    assert f.shape == (24,)
    assert np.isfinite(f).all()


def test_make_dataset():
    sets = [simulate_read_set(n_reads=50, seed=s) for s in (1, 2)]
    X, y = make_dataset(sets)
    assert X.shape == (100, 24)
    assert set(np.unique(y)) <= {0, 1}


def test_profile_matrix_shape():
    rs = simulate_read_set(n_reads=120, seed=9)
    P = profile_matrix(rs.reads, 20)
    assert P.shape == (2, 20)


# --------------------------------------------------------------------------- rl env

def test_env_api_and_episode_length():
    env = GenomeGapFillEnv(seq_len=80, gap_frac=0.25, seed=42)
    obs, info = env.reset(seed=42)
    assert obs.shape == (OBS_DIM,)
    assert info["gap_positions"]
    res = run_episode(env, RandomFiller(), seed=42)
    assert res["n_gaps"] == len(res["trace"])
    assert 0.0 <= res["accuracy"] <= 1.0


def test_env_deterministic_with_seed():
    e1 = GenomeGapFillEnv(seq_len=60, seed=7)
    e2 = GenomeGapFillEnv(seq_len=60, seed=7)
    o1, i1 = e1.reset(seed=7)
    o2, i2 = e2.reset(seed=7)
    assert np.allclose(o1, o2)
    assert i1["gap_positions"] == i2["gap_positions"]


# --------------------------------------------------------------------------- funcgen

def test_funcgen_orf_and_translate():
    from lazarus.core import funcgen as fg
    from lazarus.data.species_db import GENE_TEMPLATES
    dna = GENE_TEMPLATES["mammoth_hbb"].donor_dna
    orfs = fg.six_frame_orfs(dna, 30)
    assert any(o["aa_len"] > 100 for o in orfs)
    assert fg.codon_usage(dna) and 0 < fg.cai(dna) <= 1


def test_funcgen_protein_and_motifs():
    from lazarus.core import funcgen as fg
    props = fg.protein_props("MVHLTPEEKS")
    assert props["mw"] > 1000 and 4 < props["pi"] < 11
    assert fg.tm_helices("K" * 5 + "LIVAFLLIVAFLLIVAFLLIVAF" + "K" * 5)
    assert any(h["motif"] == "N-glycosylation" for h in fg.find_motifs("MANKTSTP"))


def test_registry_counts():
    from lazarus.registry import DIVISIONS, TOOLS, status_counts
    assert len(TOOLS) == 115
    assert len(DIVISIONS) == 10
    assert sum(status_counts().values()) == 115
    assert all(t.division in DIVISIONS for t in TOOLS)


def test_titan_tips_every_console():
    from lazarus.titan.tips import tips_for
    for key in ("home", "adna", "phylo", "rl", "crispr", "scorecard", "palette",
                "funcgen", "generic"):
        assert len(tips_for(key)) >= 3


# --------------------------------------------------------------------------- species db

def test_scores_and_ranking():
    for sp in SPECIES.values():
        sc = candidate_score(sp)
        assert 0 <= sc <= 100
    rows = ranked({"adna_quality": 1.0})
    assert rows[0][0].key in ("mammoth", "pyrenean_ibex", "thylacine")
