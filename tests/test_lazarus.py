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
    from lazarus.registry import DIVISIONS, TOOLS, status_counts, tool_by_id, tools_by_ids
    assert len(TOOLS) == 115
    assert len(DIVISIONS) == 10
    assert sum(status_counts().values()) == 115
    assert all(t.division in DIVISIONS for t in TOOLS)
    # tool_by_id() helper — the lookup the palette and status wall rely on
    t8 = tool_by_id(8)
    assert t8 is not None and t8.name == "FASTA/FASTQ QC Dashboard"
    assert t8.status == "live" and t8.console.startswith("pages/")
    assert tool_by_id(9999) is None
    assert [t.tid for t in tools_by_ids([110, 112, 113])] == [110, 112, 113]
    assert len(tools_by_ids([110, 9999, 113])) == 2
    # every live tool must be mapped to a page
    assert all(t.console for t in TOOLS if t.status == "live")


def test_titan_tips_every_console():
    from lazarus.titan.tips import tips_for
    # every console key, including the Wave-2 additions
    for key in ("home", "adna", "phylo", "rl", "crispr", "scorecard", "palette",
                "funcgen", "generic", "import", "qc", "assembly", "evolution", "ops"):
        assert len(tips_for(key)) >= 3
    assert tips_for("no-such-key") == tips_for("generic")


# --------------------------------------------------------------------------- species db

def test_scores_and_ranking():
    for sp in SPECIES.values():
        sc = candidate_score(sp)
        assert 0 <= sc <= 100
    rows = ranked({"adna_quality": 1.0})
    assert rows[0][0].key in ("mammoth", "pyrenean_ibex", "thylacine")


# ===========================================================================
# Wave 2 — QC / library prep  (#8–#13)
# ===========================================================================

def _qc_library(n: int = 220, seed: int = 3):
    return simulate_read_set(n_reads=n, frag_mean=45, damage_5p=0.30, damage_3p=0.22,
                             contamination=0.15, seed=seed).reads


def test_qc_report_fields():
    from lazarus.core import qc

    rep = qc.qc_report(_qc_library())
    assert rep.n_reads == 220
    assert 0.0 <= rep.q20_rate <= 1.0 and 0.0 <= rep.gc_content <= 1.0
    assert rep.per_pos_quality.shape == rep.per_pos_gc.shape == (qc.MAX_PROFILE_POS,)
    assert rep.simulated_quality          # no FASTQ qualities were supplied
    assert {f["check"] for f in rep.flags} >= {"Q20 fraction", "GC content", "Duplication"}

    # FASTQ qualities must be honoured when they are supplied
    quals = [np.full(len(r.seq), 38.0) for r in rep and _qc_library(40)]
    rep2 = qc.qc_report(_qc_library(40), quals=quals)
    assert not rep2.simulated_quality and rep2.q30_rate > 0.99


def test_pmd_scores_bounds():
    from lazarus.core import qc

    ancient = [r for r in _qc_library(300) if r.is_ancient]
    modern = [r for r in _qc_library(300) if not r.is_ancient]
    pa, pm = qc.pmd_scores(ancient), qc.pmd_scores(modern)
    assert pa.shape == (len(ancient),) and pm.shape == (len(modern),)
    assert (pa >= 0).all() and (pa <= 1).all()
    # deaminated molecules must score higher than undamaged contaminants
    assert pa.mean() > pm.mean()


def test_library_complexity_and_dedup():
    from lazarus.core import qc

    reads = _qc_library(200)
    duped = reads + reads[:60]                       # 60 guaranteed duplicates
    curve = qc.complexity_curve(duped, n_points=10)
    assert curve["total_reads"] == len(duped)
    assert curve["complexity_estimate"] > 0
    assert curve["distinct"] == sorted(curve["distinct"])   # monotone non-decreasing

    kept, stats = qc.deduplicate(duped)
    assert stats["mode"] == "exact"
    assert stats["removed"] == 60 and stats["kept"] == len(reads)
    assert stats["duplicate_rate"] == pytest.approx(round(60 / len(duped), 4), abs=1e-4)

    # near-duplicate collapsing is strictly more aggressive than exact matching
    near, near_stats = qc.deduplicate(duped, max_mismatches=2)
    assert near_stats["kept"] <= stats["kept"]


def test_trim_reads_adapter_and_quality():
    from lazarus.core import qc

    adapter = qc.DEFAULT_ADAPTER
    reads = [Read(seq="ACGTACGTACGTACGTACGTACGT" + adapter + "TTTT", source="",
                  pos=0, is_ancient=True)]
    trimmed, tq, stats = qc.trim_reads(reads, adapter=adapter, min_quality=0,
                                       window=4, min_length=5)
    assert stats["adapter_clipped"] == 1
    assert trimmed and adapter not in trimmed[0].seq

    # a low-quality tail must be removed and short survivors dropped
    long_read = Read(seq="A" * 80, source="A" * 80, pos=0, is_ancient=True)
    trimmed2, _, stats2 = qc.trim_reads([long_read], quals=[np.array([35.0] * 40 + [5.0] * 40)],
                                        adapter=adapter, min_quality=20, window=5, min_length=25)
    assert stats2["quality_trimmed"] == 1 and len(trimmed2[0].seq) <= 60
    _, _, stats3 = qc.trim_reads([Read(seq="ACGT", source="", pos=-1, is_ancient=True)],
                                 adapter=adapter, min_length=25)
    assert stats3["dropped_short"] == 1 and stats3["kept"] == 0


def test_udg_comparator_treatments():
    from lazarus.core import qc

    out = qc.udg_compare(_qc_library(200), max_reads=120)
    assert {r.treatment for r in out["rows"]} == {
        qc.UDG_TREATMENTS[k]["label"] for k in ("none", "partial", "full")}
    by_label = {r.treatment: r for r in out["rows"]}
    none_row = by_label[qc.UDG_TREATMENTS["none"]["label"]]
    full_row = by_label[qc.UDG_TREATMENTS["full"]["label"]]
    # full UDG strips the damage signal the authenticator depends on
    assert full_row.ct_pos0 < none_row.ct_pos0
    assert full_row.authenticity < none_row.authenticity
    assert out["n_reads_used"] == 120


# ===========================================================================
# Wave 2 — assembly studio  (#20–#23)
# ===========================================================================

def test_misregistration_detector():
    from lazarus.core import assembly

    env = GenomeGapFillEnv(seq_len=180, gap_frac=0.30, misreg=0.55, seed=21)
    rep = assembly.misregistration_from_env(env)
    assert rep.profile.shape == (180, len(assembly.LAGS))
    assert rep.best_lag.shape == (180,)
    assert 0.0 <= rep.mean_confidence <= 1.0
    # the detector must recover the simulated indel offsets far better than chance
    assert rep.detection_accuracy is not None and rep.detection_accuracy > 0.5
    assert all(z["length"] > 0 for z in rep.zones)
    assert sum(z["length"] for z in rep.zones) == rep.n_pos

    # a perfectly registered reference yields lag 0 everywhere and no shifted zones
    clean = assembly.detect_misregistration(env.true_seq, env.true_seq,
                                            np.ones(len(env.true_seq), dtype=bool))
    assert clean.shifted_fraction == 0.0


def test_consensus_callers():
    from lazarus.core import assembly

    genome = random_sequence(300, 0.42, random.Random(0))
    rng = random.Random(1)
    reads = []
    for _ in range(500):
        pos = rng.randrange(0, 300 - 40)
        seq = genome[pos: pos + 40]
        reads.append(Read(seq=seq, source=seq, pos=pos, is_ancient=True))
    out = assembly.consensus_compare(assembly.pile_reads(reads, 300), truth=genome, min_depth=3)
    assert out["n_sites"] > 200
    for name, acc in out["accuracy"].items():
        assert acc is not None and acc > 0.9, (name, acc)
    assert 0.0 <= out["discordance_rate"] <= 1.0


def test_coverage_and_dropout():
    from lazarus.core import assembly

    genome_len = 400
    reads = simulate_read_set(n_reads=600, genome_len=genome_len, frag_mean=45, seed=8).reads
    cov = assembly.coverage_profile(reads, genome_len, bin_size=25)
    assert cov["depth"].shape == (genome_len,)
    assert not cov["synthetic_placement"]           # simulated reads carry coordinates
    assert cov["mean_depth"] > 0 and 0.0 <= cov["breadth"] <= 1.0
    assert cov["depth_at_1x"] >= cov["depth_at_5x"] >= cov["depth_at_10x"]

    # a genome hole must be reported as one dropout run
    depth = np.ones(100)
    depth[40:65] = 0.0
    runs = assembly.dropout_runs(depth, min_len=10)
    assert len(runs) == 1 and runs[0]["start"] == 40 and runs[0]["length"] == 25


def test_kmer_spectrum_estimates_genome_size():
    from lazarus.core import assembly

    base = random_sequence(6000, 0.45, random.Random(4))
    rng = random.Random(5)
    contigs = [base[i: i + 1500] for i in
               (rng.randrange(0, 4501) for _ in range(10))]
    spec = assembly.kmer_spectrum(contigs, k=15)
    assert spec["distinct_kmers"] > 0 and spec["peak_coverage"] >= 1
    # definition of the GenomeScope-style estimator
    assert spec["genome_size_estimate"] == int(spec["total_kmers"] / spec["peak_coverage"])
    assert 0.4 * len(base) < spec["genome_size_estimate"] < 4 * len(base)
    assert 0.0 <= spec["repeat_fraction"] <= 1.0
    assert 0.0 <= spec["heterozygosity_rate"] <= 1.0


# ===========================================================================
# Wave 2 — evolution workbench  (#33, #34)
# ===========================================================================

def test_newick_surgery_parse_reroot_prune():
    from lazarus.core import evolution as evo

    tree = evo.parse_newick(evo.DEMO_NEWICK)
    leaves = evo.leaves(tree)
    assert len(leaves) == 12 and "Mammuthus_primigenius" in leaves

    # parse → export → parse must be stable
    again = evo.parse_newick(evo.to_newick(tree))
    assert sorted(evo.leaves(again)) == sorted(leaves)

    # wrapper-root rerooting keeps every tip and yields a bifurcating new root
    rero = evo.reroot(tree, "Mammuthus_primigenius")
    assert sorted(evo.leaves(rero)) == sorted(leaves)
    assert len(rero.children) == 2
    assert any(c.name == "Mammuthus_primigenius" for c in rero.children)
    assert sorted(evo.leaves(tree)) == sorted(leaves)      # original untouched
    with pytest.raises(ValueError):
        evo.reroot(tree, "Not_in_this_tree")

    keep = {"Homo_sapiens", "Pan_troglodytes", "Mammuthus_primigenius"}
    pruned = evo.prune(tree, keep)
    assert pruned is not None and set(evo.leaves(pruned)) == keep
    assert len(evo.leaves(evo.ladderize(tree))) == 12
    assert evo.tree_stats(tree)["n_leaves"] == 12


def test_substitution_model_corrections():
    from lazarus.core import evolution as evo

    seq = random_sequence(4000, 0.42, random.Random(0))
    shallow = evo.evolve(seq, "JC69", 0.10, seed=1)
    p = evo.p_distance(seq, shallow)
    jc = evo.jc69_distance(seq, shallow)
    assert 0 < p < 0.75
    assert jc > p                                  # correction can only add hidden hits
    assert jc == pytest.approx(0.10, abs=0.05)     # …and should recover the true branch

    # saturation: the gap between observed and corrected widens with time
    deep = evo.evolve(seq, "JC69", 0.80, seed=2)
    assert evo.p_distance(seq, deep) < evo.jc69_distance(seq, deep)

    # K2P separates transitions from transversions
    k2p_seq = evo.evolve(seq, "K2P", 0.20, kappa=4.0, seed=3)
    ts, tv = evo.count_ts_tv(seq, k2p_seq)
    assert ts > tv                                 # κ = 4 biases toward transitions
    assert evo.k2p_distance(seq, k2p_seq) >= evo.p_distance(seq, k2p_seq)
    assert evo.hky85_distance(seq, k2p_seq) > 0

    curve = evo.divergence_curve(seq, "HKY85",
                                 times=(0.05, 0.25, 0.60), kappa=2.5, seed=4)
    assert curve["p_distance"] == sorted(curve["p_distance"])
    assert all(v is not None for v in curve["jc69"])


# ===========================================================================
# Wave 2 — wet-lab & ops engines  (#67, #68, #88–#90)
# ===========================================================================

def test_ssodn_hdr_builder():
    from lazarus.core import wetlab

    tpl = GENE_TEMPLATES["mammoth_hbb"]
    plan = crispr.plan_edits(tpl.donor_dna, tpl.target_dna, gene_name=tpl.gene_name)
    site = plan.sites[0]
    ss = wetlab.build_ssodn(tpl.donor_dna, site.edit, arm_len=45)

    assert ss["sense"] and ss["antisense"] == wetlab.rc(ss["sense"])
    assert ss["arm_len_left"] == ss["arm_len_right"] == 45
    assert ss["total_len"] == 91
    # every requested nucleotide change must be baked into the donor
    for off, _old, new in site.edit.nt_changes:
        pos = 3 * site.edit.codon_index + off - (site.edit.dna_pos_center - 45)
        assert ss["sense"][pos] == new
    assert 40 < ss["tm_sense"] < 100
    assert isinstance(ss["blocking_options"], list) and isinstance(ss["warnings"], list)


def test_guide_cascade():
    from lazarus.core import wetlab

    tpl = GENE_TEMPLATES["mammoth_hbb"]
    cas = wetlab.guide_cascade(tpl.donor_dna, tpl.target_dna, window=90)
    assert cas["n_edits"] == 3 and cas["n_hubs"] >= 1
    assert cas["n_guides"] <= cas["n_hubs"]
    assert 0.0 <= cas["coverage"] <= 1.0
    assert sorted(cas["order"]) == list(range(1, cas["n_hubs"] + 1))
    assert len(cas["rows"]) == cas["n_hubs"]
    # a wider hub window can only merge hubs, never split them
    wide = wetlab.guide_cascade(tpl.donor_dna, tpl.target_dna, window=400)
    assert wide["n_hubs"] <= cas["n_hubs"]
    with pytest.raises(ValueError):
        wetlab.guide_cascade(tpl.donor_dna, tpl.donor_dna)


def test_ops_consoles_routes_surrogates_ethics():
    from lazarus.core import wetlab

    sp = SPECIES["mammoth"]
    routes = wetlab.route_comparator(sp)
    assert len(routes["rows"]) == 5
    assert routes["rows"][0]["score"] >= routes["rows"][-1]["score"]     # sorted
    assert routes["best"] == routes["rows"][0]["route"]
    for row in routes["rows"]:
        assert 0 <= row["score"] <= 100

    sur = wetlab.surrogate_matchmaker(sp)
    assert sur["rows"][0]["score"] >= sur["rows"][-1]["score"]
    assert sur["best"] == sur["rows"][0]["surrogate"]

    eth = wetlab.ethics_checklist(sp)
    assert 0 <= eth["overall"] <= 100 and set(eth["domains"]) >= {"welfare", "ecology", "consent"}
    seeded = eth["overall"]
    # a maximally pessimistic audit must score below the seeded baseline
    worst = {i["id"]: 0 for d in eth["domains"].values() for i in d["items"]}
    best = {i["id"]: 5 for d in eth["domains"].values() for i in d["items"]}
    assert (wetlab.ethics_score_from_ratings(eth["domains"], worst)["overall"]
            < seeded
            < wetlab.ethics_score_from_ratings(eth["domains"], best)["overall"])


# ===========================================================================
# Wave 2 — importers, LIMS, ops guard  (#105–#107, #110)
# ===========================================================================

def _csv_bytes(rows: list[list[str]]) -> bytes:
    return ("\n".join(",".join(r) for r in rows) + "\n").encode()


def test_importers_csv_fasta_fastq_and_pdf():
    from lazarus.data.importers import import_bytes, import_file, import_text

    tpl = GENE_TEMPLATES["dodo_mc1r"]

    # --- paired CSV: donor/target columns are detected and exposed in meta ---
    res = import_bytes(_csv_bytes([["gene", "donor_cds", "target_cds"],
                                   [tpl.gene_name, tpl.donor_dna, tpl.target_dna]]),
                       "pair.csv")
    assert res.paired is True
    assert res.meta["donor_sequence"] == tpl.donor_dna.upper()
    assert res.meta["target_sequence"] == tpl.target_dna.upper()

    # --- paired FASTA ---
    fa = f">donor {tpl.donor_species}\n{tpl.donor_dna}\n>target ancestral\n{tpl.target_dna}\n"
    res_fa = import_bytes(fa.encode(), "pair.fasta")
    assert res_fa.paired and res_fa.n == 2

    # --- FASTQ keeps per-base Phred in .quals ---
    seq = tpl.donor_dna[:60]
    fq = f"@r1\n{seq}\n+\n{'I' * len(seq)}\n"
    res_fq = import_bytes(fq.encode(), "r.fastq")
    assert res_fq.n == 1 and len(res_fq.quals) == 1
    assert res_fq.quals[0].tolist() == [float(ord("I") - 33)] * len(seq)

    # --- read-set CSV is NOT treated as a pair ---
    res_reads = import_bytes(_csv_bytes([["read_id", "sequence"]] +
                                        [[f"r{i}", r.seq] for i, r in
                                         enumerate(simulate_read_set(n_reads=5, seed=1).reads)]),
                             "reads.csv")
    assert res_reads.n == 5 and res_reads.paired is False
    assert res_reads.summary()["reads"] == 5

    # --- empty / unparseable input degrades gracefully ---
    empty = import_text("", "empty")
    assert empty.n == 0 and empty.notes

    # --- import_file accepts a path ---
    import tempfile
    with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as fh:
        fh.write(_csv_bytes([["donor_cds", "target_cds"],
                             [tpl.donor_dna, tpl.target_dna]]))
        path = fh.name
    assert import_file(path).paired is True

    # --- PDF: a hand-rolled document must yield its embedded sequences + the pair ---
    import zlib
    from lazarus.data.importers import import_bytes as _import_bytes

    seqs = [random_sequence(90, 0.42, random.Random(i)) for i in range(3)]
    lines = ["Project Lazarus supplementary methods",
             "donor: " + seqs[0],
             "target: " + seqs[1],
             "read: " + seqs[2]]
    content = "\n".join(["BT", "/F1 9 Tf", "72 740 Td", "14 TL"]
                        + [f"({ln}) Tj T*" for ln in lines] + ["ET"]).encode("latin-1")
    stream = zlib.compress(content)
    objs = [b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
            b"/Resources << /Font << /F1 5 0 R >> >> >>",
            b"<< /Filter /FlateDecode /Length " + str(len(stream)).encode() + b" >>\nstream\n"
            + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode() + b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref}\n%%EOF\n").encode()

    pdf = _import_bytes(bytes(out), "methods.pdf")
    assert pdf.meta.get("format") == "pdf"
    assert pdf.n == 3 and pdf.paired is True
    assert pdf.meta["donor_sequence"] == seqs[0]
    assert pdf.meta["target_sequence"] == seqs[1]


def test_lims_chain_of_custody(tmp_path):
    from lazarus.data import lims

    db = tmp_path / "lims.sqlite"
    lims.init_db(db)
    assert lims.stats(db)["specimens"] == 0

    spec = lims.register_specimen("LZ-TEST", "Mammuthus primigenius", "Woolly mammoth",
                                  origin="Wrangel Island", collected_by="field team",
                                  source_format="fastq", n_reads=1200, path=db)
    assert spec["id"] == "LZ-TEST"
    # registering logs the accessioning event by itself
    assert lims.chain_of_custody("LZ-TEST", db)[0]["action"] == "accessioned"

    lims.log_event("LZ-TEST", "extracted", actor="Lab 2", location="clean room", path=db)
    lims.register_library("LZ-TEST-L1", "LZ-TEST", n_reads=1200, mean_len=44.5, path=db)

    chain = lims.chain_of_custody("LZ-TEST", db)
    assert [e["action"] for e in chain] == ["accessioned", "extracted", "library_prepped"]
    assert lims.custody_integrity("LZ-TEST", db)["ok"] is True
    assert lims.list_libraries("LZ-TEST", db)[0]["mean_len"] == 44.5
    assert "LZ-TEST" in lims.export_json(db)

    # bridged straight from an ImportResult
    from lazarus.data.importers import ImportResult
    imp = ImportResult(reads=[Read(seq="ACGTACGTACGTAC", source="", pos=-1, is_ancient=True)],
                       notes=["unit test"], meta={"format": "fasta", "source": "test"})
    lims.import_result_to_lims(imp, "LZ-BRIDGE", taxon="test", path=db)
    assert lims.get_specimen("LZ-BRIDGE", db)["n_reads"] == 1

    assert lims.delete_specimen("LZ-TEST", db) is True
    assert lims.get_specimen("LZ-TEST", db) is None
    assert lims.custody_integrity("LZ-TEST", db)["problems"]


def test_ops_locker_never_touches_shipped_weights():
    from lazarus.config import MODELS_DIR, WEIGHTS_DQN
    from lazarus.core import ops

    assert ops.locker_dir("demo") == MODELS_DIR / "locker_demo"
    # every shipped artefact is refused…
    for p in ops.SHIPPED_WEIGHTS:
        with pytest.raises(ValueError):
            ops.assert_not_shipped(p)
    # …and so is any other file dropped straight into models/
    with pytest.raises(ValueError):
        ops.assert_not_shipped(MODELS_DIR / "dqn_gapfill.pt")
    with pytest.raises(ValueError):
        ops.assert_not_shipped(MODELS_DIR / "read_authenticity_mlp.pt")
    # a locker path is fine
    assert ops.assert_not_shipped(ops.locker_dir("demo") / "weights" / "dqn_gapfill.pt").exists() is False

    integrity = ops.registry_integrity()
    assert integrity["ok"], integrity["problems"]
    assert integrity["counts"]["total"] if "total" in integrity["counts"] else True

    status = ops.repo_status()
    assert status["tools"]["total"] == 115 and status["n_pages"] >= 7
    assert status["headline_metrics"]["mlp_accuracy"] == pytest.approx(0.8963)

    payload = ops.report_payload("t", [{"heading": "h", "kind": "kv", "data": {"a": 1}}])
    assert '"a": 1' in ops.to_json(payload)
    assert "key,value" in ops.to_csv(payload)
    assert "<h1>t</h1>" in ops.to_html(payload)
    assert WEIGHTS_DQN.exists()          # nothing above may have removed it
