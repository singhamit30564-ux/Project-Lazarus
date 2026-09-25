"""Master tool registry — 115 tools across 10 divisions.

Status vocabulary:
  live    — shipped and interactive in the app
  wave2   — being built now (Division consoles arriving next)
  roadmap — queued for later waves
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolSpec:
    tid: int
    name: str
    division: str      # division id: "I".."X"
    status: str        # live | wave2 | roadmap
    blurb: str
    engine: str = ""   # "ML", "RL", "ML+RL", "engine tag" or ""
    console: str = ""  # streamlit page file where it lives ("" if not yet)


DIVISIONS: dict[str, dict] = {
    "I":   {"name": "Paleogenomics Lab", "emoji": "🧬", "blurb": "Authentication, damage chemistry, contamination, fragmentomics."},
    "II":  {"name": "Genome Reconstruction", "emoji": "🧩", "blurb": "Gap filling, scaffolding, consensus, RL policy studio."},
    "III": {"name": "Phylogenomics", "emoji": "🌳", "blurb": "Placement, trees, distances, ancestry, selection."},
    "IV":  {"name": "Functional Genomics", "emoji": "🧫", "blurb": "ORFs, translation, codons, protein biophysics, motifs."},
    "V":   {"name": "Extinct Protein Studio", "emoji": "❄️", "blurb": "Resurrection proteins: hemoglobin, stability, interfaces."},
    "VI":  {"name": "Genome Engineering", "emoji": "✂️", "blurb": "CRISPR planning: guides, prime editing, HDR, safeguards."},
    "VII": {"name": "Synthetic Genomics", "emoji": "🧵", "blurb": "Genome composition, repeats, telomeres, chromosomes."},
    "VIII": {"name": "Program Ops", "emoji": "🗺️", "blurb": "Candidate strategy, surrogates, timelines, ethics, budgets."},
    "IX":  {"name": "Cryo & Cell Bio", "emoji": "🐄", "blurb": "iPSC, SCNT, artificial wombs, germline routes."},
    "X":   {"name": "Data Ops & Platform", "emoji": "🛠️", "blurb": "Formats, LIMS, metadata, training consoles, reports."},
}

_P1 = "pages/1_🧬_Ancient_DNA_Analyzer.py"
_P2 = "pages/2_🌳_Phylogenetic_Placement.py"
_P3 = "pages/3_🤖_Genome_Reconstruction_RL.py"
_P4 = "pages/4_✂️_CRISPR_Edit_Planner.py"
_P5 = "pages/5_🦣_Candidate_Scorecard.py"
_P6 = "pages/6_🧫_Functional_Genomics.py"
_P0 = "pages/0_🔭_Tool_Palette.py"
_P7 = "pages/7_📥_Import_Studio.py"
_P8 = "pages/8_🧪_QC_&_Library_Prep.py"
_P9 = "pages/9_🧩_Assembly_Studio.py"
_P10 = "pages/10_🌿_Evolution_Workbench.py"
_P11 = "pages/11_🛠️_Training_&_Ops.py"

TOOLS: tuple[ToolSpec, ...] = (
    # ---------------- I — Paleogenomics Lab ----------------
    ToolSpec(1,  "Ancient DNA Authenticator", "I", "live", "One-call authentication verdict on any library.", "", _P1),
    ToolSpec(2,  "Damage Parameter Fitter", "I", "live", "Briggs-style δS / δD / λ estimation from misincorporation curves.", "", _P1),
    ToolSpec(3,  "Terminal Misincorporation Profiler", "I", "live", "C→T 5′ / G→A 3′ position curves, mapDamage-style.", "", _P1),
    ToolSpec(4,  "Fragment Length Forensics", "I", "live", "Length distribution, quantiles, ancient-vs-modern shape tests.", "", _P1),
    ToolSpec(5,  "Contamination Estimator", "I", "live", "ML + heuristic modern-DNA contamination estimates.", "ML", _P1),
    ToolSpec(6,  "Read Authenticity Classifier", "I", "live", "Per-read ancient-vs-modern probabilities (24-dim fragmentomics MLP).", "ML", _P1),
    ToolSpec(7,  "Damage Tier Classifier", "I", "live", "Library-level damage tier from curves (1-D CNN).", "ML", _P1),
    ToolSpec(8,  "FASTA/FASTQ QC Dashboard", "I", "live", "FastQC-style per-base quality & composition dashboard.", "", _P8),
    ToolSpec(9,  "PMD-Score Calculator", "I", "live", "Per-read posterior probability of methylation damage.", "ML", _P8),
    ToolSpec(10, "Library Complexity Estimator", "I", "live", "Preseq-style duplication & complexity curves.", "", _P8),
    ToolSpec(11, "Duplicate Read Remover", "I", "live", "Exact/near duplicate collapse with UMI support.", "", _P8),
    ToolSpec(12, "Adapter & Quality Trimmer", "I", "live", "Sliding-window trimming + adapter clipping.", "", _P8),
    ToolSpec(13, "UDG-Treatment Comparator", "I", "live", "Partial/full UDG effect on damage authentication.", "", _P8),
    ToolSpec(14, "Molecular-Clock Age Estimator", "I", "roadmap", "Damage-accumulation age intuition from δD decay.", "", ""),
    ToolSpec(15, "End-Repair Simulator", "I", "roadmap", "Simulate end-repair/blunt/overhang library preps.", "", ""),

    # ---------------- II — Genome Reconstruction ----------------
    ToolSpec(16, "Genome Gap-Fill RL Console", "II", "live", "GenomeGapFill-v0 worlds, DQN agent fills every gap.", "RL", _P3),
    ToolSpec(17, "Policy Leaderboard", "II", "live", "DQN vs registration-rule / greedy / pileup / quality / random.", "RL", _P3),
    ToolSpec(18, "Episode Walkthrough", "II", "live", "Step through every base-call decision with truth trace.", "RL", _P3),
    ToolSpec(19, "DQN Training Console", "II", "live", "Curves, checkpoints, warm-start, in-app retraining.", "RL", _P3),
    ToolSpec(20, "Misregistration Detector", "II", "live", "Lag-agreement profile across reference offsets (indel audit).", "", _P9),
    ToolSpec(21, "Consensus Caller Studio", "II", "live", "Majority / Bayesian / PMD-aware consensus comparison.", "", _P9),
    ToolSpec(22, "Coverage Depth Analyzer", "II", "live", "Per-base coverage, breadth/depth, dropout maps.", "", _P9),
    ToolSpec(23, "K-mer GenomeScope", "II", "live", "Genome size, heterozygosity & repeat spectra from k-mers.", "", _P9),
    ToolSpec(24, "Reference Bias Detector", "II", "roadmap", "Allele-balance shifts toward the reference allele.", "", ""),
    ToolSpec(25, "Reference-Guided Scaffolder", "II", "roadmap", "Order/orient contigs against a misregistration-aware reference.", "", ""),
    ToolSpec(26, "Gap Mapper & N-Atlas", "II", "roadmap", "Gap-size distribution & N-content cartography.", "", ""),
    ToolSpec(27, "Assembly Merger", "II", "roadmap", "Quick-merge style consensus of two assemblies.", "", ""),
    ToolSpec(28, "Pseudo-haplotype Resolver", "II", "roadmap", "Separate damaged-heterozygous from damaged-homozygous sites.", "", ""),

    # ---------------- III — Phylogenomics ----------------
    ToolSpec(29, "Phylogenetic Placement Console", "III", "live", "Place an ancient query among candidate relatives.", "", _P2),
    ToolSpec(30, "k-mer Distance Matrix Explorer", "III", "live", "Mash-style distances + heatmap.", "", _P2),
    ToolSpec(31, "Neighbor-Joining Tree Builder", "III", "live", "NJ cladogram with branch lengths.", "", _P2),
    ToolSpec(32, "Classical MDS Embedding", "III", "live", "Distance geometry scatter of the panel.", "", _P2),
    ToolSpec(33, "Newick Viewer / Editor", "III", "live", "Parse, reroot, prune and export Newick trees.", "", _P10),
    ToolSpec(34, "Sequence Divergence Simulator", "III", "live", "Jukes-Cantor / Kimura / HKY evolution playground.", "", _P10),
    ToolSpec(35, "Ancestral Sequence Reconstructor", "III", "roadmap", "Marginal reconstruction of internal node sequences.", "ML", ""),
    ToolSpec(36, "Divergence-Time Estimator", "III", "roadmap", "Relaxed-clock toy TMRCA estimates.", "", ""),
    ToolSpec(37, "Introgression Detector (D-stat)", "III", "roadmap", "ABBA-BABA statistics on biallelic sites.", "", ""),
    ToolSpec(38, "dN/dS Selection Calculator", "III", "roadmap", "Synonymous vs nonsynonymous divergence.", "", ""),
    ToolSpec(39, "Ortholog Finder (RBH)", "III", "roadmap", "Reciprocal-best-hit ortholog tables.", "", ""),
    ToolSpec(40, "Synteny Dot-Plotter", "III", "roadmap", "Genome-to-genome collinearity plots.", "", ""),

    # ---------------- IV — Functional Genomics ----------------
    ToolSpec(41, "Gene / ORF Finder", "IV", "wave2", "Six-frame ORF scan with length filters.", "", _P6),
    ToolSpec(42, "Translation Workbench", "IV", "wave2", "Frame-aware codon translation & stop mapping.", "", _P6),
    ToolSpec(43, "Codon Usage Analyzer", "IV", "wave2", "Codon counts, RSCU, CAI vs host preference.", "", _P6),
    ToolSpec(44, "Protein Property Calculator", "IV", "wave2", "MW, pI, GRAVY, aromaticity, instability index.", "", _P6),
    ToolSpec(45, "Hydropathy & TM-Helix Predictor", "IV", "wave2", "Kyte-Doolittle profile + TM calls.", "", _P6),
    ToolSpec(46, "Motif & Active-Site Scanner", "IV", "wave2", "PROSITE-style patterns + custom regex.", "", _P6),
    ToolSpec(47, "CpG Island Finder", "IV", "wave2", "GC/obs-exp CpG island detection.", "", _P6),
    ToolSpec(48, "Promoter Element Scanner", "IV", "roadmap", "TATA/CAAT/GC-box element grammar.", "", ""),
    ToolSpec(49, "Host Codon Optimizer", "IV", "roadmap", "Rewrite CDS for elephant / dunnart / pigeon hosts.", "", ""),
    ToolSpec(50, "Signal Peptide Screener", "IV", "roadmap", "n-h/c-h region signal peptide heuristics.", "", ""),
    ToolSpec(51, "Secondary Structure Sketcher", "IV", "roadmap", "Chou-Fasman helix/sheet/turn propensities.", "", ""),
    ToolSpec(52, "Sequence Logo Builder", "IV", "roadmap", "Motif conservation logos.", "", ""),

    # ---------------- V — Extinct Protein Studio ----------------
    ToolSpec(53, "Ancestral Protein Designer", "V", "wave3", "Resurrection mutation sets from consensus reconstruction.", "ML", ""),
    ToolSpec(54, "Hemoglobin O₂-Affinity Predictor", "V", "wave3", "Thermal-adapted Hb oxygen curves (mammoth Hb story).", "ML", ""),
    ToolSpec(55, "Thermal Stability Index", "V", "wave3", "Thermoadaptation score from composition & motifs.", "", ""),
    ToolSpec(56, "Oligomerization Interface Mapper", "V", "wave3", "Interface conservation across resurrected variants.", "", ""),
    ToolSpec(57, "Ligand Pocket Detector", "V", "roadmap", "Cavity & binding-motif heuristics.", "", ""),
    ToolSpec(58, "ΔΔG Stability Estimator", "V", "roadmap", "Mutation stability deltas (Miyazawa-Jernigan toy).", "ML", ""),
    ToolSpec(59, "Epitope Conservancy Checker", "V", "roadmap", "Antigenic site conservation vs host immune system.", "", ""),
    ToolSpec(60, "Immunogenicity / Allergen Flag", "V", "roadmap", "Allergen-like motif screens for revived proteins.", "ML", ""),
    ToolSpec(61, "Moonlighting Function Predictor", "V", "roadmap", "Multi-function signals (crystallins, lectins…).", "ML", ""),
    ToolSpec(62, "Enzyme Efficiency Estimator", "V", "roadmap", "kcat/Km intuition from triad & pocket context.", "", ""),

    # ---------------- VI — Genome Engineering ----------------
    ToolSpec(63, "CRISPR Edit Planner", "VI", "live", "Codon edit tables resurrecting ancestral alleles.", "", _P4),
    ToolSpec(64, "SpCas9 Guide Designer", "VI", "live", "20-nt spacers + NGG PAM + position scoring.", "", _P4),
    ToolSpec(65, "Off-Target Seed Scanner", "VI", "live", "PAM-proximal seed repeat risk heuristics.", "", _P4),
    ToolSpec(66, "Prime-Edit pegRNA Designer", "VI", "live", "Nick guide + RTT/PBS fallback sketches.", "", _P4),
    ToolSpec(67, "ssODN HDR Template Builder", "VI", "live", "Donor designs with 40+ bp arms, both strands.", "", _P4),
    ToolSpec(68, "Whole-CDS Guide Cascade", "VI", "live", "Multiplex edit sets across large coding targets.", "", _P4),
    ToolSpec(69, "Base Editor Planner", "VI", "roadmap", "CBE/ABE windows & bystander checks.", "", ""),
    ToolSpec(70, "Multiplex Edit Assembler", "VI", "roadmap", "Edit-order & phasing strategy for cascades.", "RL", ""),
    ToolSpec(71, "PAM Atlas", "VI", "roadmap", "SaCas9/Cas12/CasX PAM landscape explorer.", "", ""),
    ToolSpec(72, "Recombineering Planner", "VI", "roadmap", "BAC-scale edits & landing pads.", "", ""),
    ToolSpec(73, "Synthetic Chromosome Assembler", "VI", "roadmap", "Chromosome-scale design & SCRaMbLE-style toggles.", "", ""),
    ToolSpec(74, "Gene-Drive Safeguard Designer", "VI", "roadmap", "Terminator switches & reversal drives.", "", ""),

    # ---------------- VII — Synthetic Genomics ----------------
    ToolSpec(75, "Genome Composer", "VII", "wave3", "FASTA assembly & sequence artboard.", "", ""),
    ToolSpec(76, "Repeat Masker", "VII", "wave3", "Tandem & interspersed repeat detection.", "", ""),
    ToolSpec(77, "GC Isochore Mapper", "VII", "wave3", "GC domain cartography.", "", ""),
    ToolSpec(78, "Transposon Detector", "VII", "wave3", "DNA/LINE/SINE signature screens.", "", ""),
    ToolSpec(79, "Telomere Motif Finder", "VII", "wave3", "TTAGGG-variant terminal repeats.", "", ""),
    ToolSpec(80, "Centromere Satellite Sketcher", "VII", "wave3", "Alpha-satellite higher-order repeat sketches.", "", ""),
    ToolSpec(81, "mtDNA Annotator", "VII", "wave3", "Mitochondrial genes, GC-skew origins.", "", ""),
    ToolSpec(82, "Y-Chromosome Gap Predictor", "VII", "roadmap", "X-transposed / ampliconic hard-to-recover regions.", "", ""),
    ToolSpec(83, "Segmental Duplication Finder", "VII", "roadmap", "Recent duplication & gene conversion tracts.", "", ""),
    ToolSpec(84, "Minimal Genome Reducer", "VII", "roadmap", "Core-gene minimization planner.", "RL", ""),

    # ---------------- VIII — Program Ops ----------------
    ToolSpec(85, "Candidate Scorecard", "VIII", "live", "Weighted revival-feasibility ranking (8 species).", "", _P5),
    ToolSpec(86, "Factor Weight Studio", "VIII", "live", "Live multi-criteria weight exploration.", "", _P5),
    ToolSpec(87, "Species Dossier Browser", "VIII", "live", "Extinction stories, relatives, routes, field notes.", "", _P5),
    ToolSpec(88, "Revival Route Comparator", "VIII", "live", "Cloning vs editing vs back-breeding decision matrix.", "", _P5),
    ToolSpec(89, "Surrogate Matchmaker", "VIII", "live", "Gestation/pouch/cycle fit scoring.", "", _P5),
    ToolSpec(90, "Ethics Review Checklist", "VIII", "live", "Interactive welfare/biosafety/social-license audit.", "", _P5),
    ToolSpec(91, "Genetic Load Calculator", "VIII", "roadmap", "Inbreeding depression & deleterious load.", "", ""),
    ToolSpec(92, "Genetic Rescue Planner", "VIII", "roadmap", "Adaptive introgression candidate tracts.", "ML", ""),
    ToolSpec(93, "Population Viability Simulator", "VIII", "roadmap", "Founding herd PVA with Allee effects.", "RL", ""),
    ToolSpec(94, "Timeline / Gantt Planner", "VIII", "roadmap", "Multi-year revival program scheduling.", "", ""),
    ToolSpec(95, "Program Cost Estimator", "VIII", "roadmap", "Budget model from biobank to rewilding.", "", ""),
    ToolSpec(96, "Habitat Readiness Index", "VIII", "roadmap", "Niche availability & ecosystem function fit.", "ML", ""),

    # ---------------- IX — Cryo & Cell Bio ----------------
    ToolSpec(97,  "Cryopreservation Viability Calculator", "IX", "wave4", "Freeze/thaw survival & ice-formation risk.", "", ""),
    ToolSpec(98,  "iPSC Reprogramming Planner", "IX", "wave4", "Factor cocktails, passage budgets, karyotyping gates.", "", ""),
    ToolSpec(99,  "SCNT Protocol Designer", "IX", "wave4", "Nuclear transfer timing & activation schedules.", "", ""),
    ToolSpec(100, "Artificial Womb Parameter Sheet", "IX", "wave4", "Biobag-style perfusion & oxygenation targets.", "", ""),
    ToolSpec(101, "Germline Route Selector", "IX", "wave4", "PGC vs somatic editing decision guide.", "", ""),
    ToolSpec(102, "Cloning Success Bayesian", "IX", "wave4", "Success posteriors seeded with the Celia prior.", "ML", ""),
    ToolSpec(103, "Chimerism Risk Assessor", "IX", "roadmap", "Contribution & germline-transmission odds.", "", ""),
    ToolSpec(104, "Surrogate Cycle Scheduler", "IX", "roadmap", "Estrous/pouch synchronization planner.", "", ""),

    # ---------------- X — Data Ops & Platform ----------------
    ToolSpec(105, "Sequence Format Converter", "X", "live", "FASTA/FASTQ/CSV/NEXUS conversions.", "", _P7),
    ToolSpec(106, "Specimen LIMS", "X", "live", "Sample registry with provenance fields.", "", _P7),
    ToolSpec(107, "Chain-of-Custody Tracker", "X", "live", "Permafrost-to-sequencer audit trail.", "", _P7),
    ToolSpec(108, "MIxS Metadata Builder", "X", "wave4", "Genomic standards checklists & export.", "", ""),
    ToolSpec(109, "Batch Pipeline Runner", "X", "wave4", "Chain tools into reproducible workflows.", "", ""),
    ToolSpec(110, "ML Training Console", "X", "live", "Train/eval supervised models from the UI.", "ML", _P11),
    ToolSpec(111, "RL Training Console", "X", "live", "DQN training, warm-start & benchmarks.", "RL", _P3),
    ToolSpec(112, "Benchmark Suite Runner", "X", "live", "Versioned evals with shipped checkpoints.", "ML+RL", _P11),
    ToolSpec(113, "Report Generator", "X", "live", "JSON/CSV/HTML run reports.", "", _P11),
    ToolSpec(114, "Tool Palette & Roadmap", "X", "live", "Searchable index of all 115 tools.", "", _P0),
    ToolSpec(115, "Dr. Titan Advisor", "X", "live", "Field notes & science counsel on every console.", "", "all pages"),
)


def tools_by_division() -> dict[str, list[ToolSpec]]:
    out: dict[str, list[ToolSpec]] = {k: [] for k in DIVISIONS}
    for t in TOOLS:
        out[t.division].append(t)
    return out


def status_counts() -> dict[str, int]:
    counts = {"live": 0, "wave2": 0, "wave3": 0, "wave4": 0, "roadmap": 0}
    for t in TOOLS:
        counts[t.status] = counts.get(t.status, 0) + 1
    return counts


_BY_ID: dict[int, ToolSpec] = {t.tid: t for t in TOOLS}


def tool_by_id(tid: int) -> ToolSpec | None:
    """Look up a tool by its registry id (1..115), or None if it is not registered."""
    return _BY_ID.get(int(tid))


def tools_by_ids(tids) -> list[ToolSpec]:
    """Batch lookup preserving the requested order and skipping unknown ids."""
    return [t for t in (tool_by_id(i) for i in tids) if t is not None]


def search_tools(query: str, division: str | None = None, status: str | None = None) -> list[ToolSpec]:
    q = query.strip().lower()
    out = []
    for t in TOOLS:
        if division and t.division != division:
            continue
        if status and t.status != status:
            continue
        if q and q not in f"{t.name} {t.blurb} {t.engine}".lower():
            continue
        out.append(t)
    return out
