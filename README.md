# 🦴 Project Lazarus

**De-Extinction Intelligence Suite** — a Streamlit + ML/RL workbench for bringing the dead back to life (digitally first).

Authenticate ancient DNA. Place a specimen on the tree of life. Reconstruct its shattered genome with a reinforcement-learning agent. Plan the CRISPR edits that re-wire a living relative toward the lost ancestor. Rank which species should walk the Earth again.

---

## ✨ The twelve consoles

| # | Console | What it does | Engine |
|---|---------|--------------|--------|
| 🧬 | **Ancient DNA Analyzer** | mapDamage-style terminal misincorporation profiles (C→T at 5′, G→A at 3′), Briggs damage-parameter fitting (δ<sub>S</sub>, δ<sub>D</sub>, λ), fragment-length fragmentomics, contamination estimate, 0–100 authenticity verdict · **CSV/PDF/FASTA upload** | damage chemistry model + read-authenticity MLP + damage-profile CNN |
| 🌳 | **Phylogenetic Placement** | Mash-style k-mer distances → neighbor-joining cladogram → classical MDS embedding of an ancient query among candidate relatives | k-mer statistics + NJ |
| 🤖 | **Genome Reconstruction RL** | `GenomeGapFill-v0` Gymnasium env: the agent fills every gap of a degraded genome by re-registering a misindel-shifted relative reference against noisy read pileups | DQN vs registration-rule / reference-greedy / pileup-greedy / quality-aware / random baselines |
| ✂️ | **CRISPR Edit Planner** | codon-level edit table that resurrects ancestral proteins in the relative's genome; SpCas9 guide design with off-target seed heuristics; prime-editing pegRNA fallback · **#67 ssODN HDR builder**, **#68 whole-CDS guide cascade**, paired-file upload | guide scoring heuristics |
| 🦣 | **Candidate Scorecard** | weighted multi-criteria feasibility ranking of real candidates (mammoth, thylacine, dodo, great auk, bucardo, Steller's sea cow, Christmas Island rat, Irish elk) · **#88 route comparator**, **#89 surrogate matchmaker**, **#90 ethics checklist** | decision model |
| 🧫 | **Functional Genomics** | ORFs, translation, codon usage/CAI, protein biophysics, Kyte–Doolittle hydropathy + TM helices, PROSITE-style motifs, CpG islands · upload mode | sequence analytics |
| 📥 | **Import Studio** | ingests **CSV / TSV / FASTA / FASTQ / JSON / PDF** → reads or a donor/target pair; **#105 format converter**, **#106 specimen LIMS**, **#107 chain-of-custody tracker** (SQLite) | importers + LIMS |
| 🧪 | **QC & Library Prep** | **#8** FastQC-style per-base dashboard, **#9** PMD posteriors, **#10** preseq-style complexity, **#11** duplicate collapse, **#12** adapter/quality trimming, **#13** UDG comparator | fragmentomics |
| 🧩 | **Assembly Studio** | **#20** misregistration detector (lag-agreement audit), **#21** majority / Bayesian / PMD-aware consensus, **#22** coverage & dropout cartography, **#23** k-mer GenomeScope | pileup + k-mer statistics |
| 🌿 | **Evolution Workbench** | **#33** Newick viewer/editor (parse, reroot, prune, ladderise, NEXUS export), **#34** JC69 / K2P / HKY85 divergence simulator with saturation curves | substitution models |
| 🛠️ | **Training & Ops** | **#110** ML/RL retrain lockers, **#112** nightshift rota + versioned benchmark suite, **#113** repo status wall + JSON/CSV/HTML reports | ML + RL + ops |
| 🔭 | **Tool Palette** | searchable index of all **115 tools** across 10 divisions, with live/wave/roadmap status | registry |

> **Retrains never touch the shipped weights.** Every run in the Training & Ops console writes
> into its own `models/locker_<tag>/` directory; `ops.assert_not_shipped()` raises if anything
> tries to write a shipped checkpoint (or any other file directly into `models/`).

## 🚀 Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# train the models once (CPU, ~2 min total)
python -m lazarus.ml.train     # read-authenticity MLP + damage CNN
python -m lazarus.rl.train     # gap-fill DQN (1200 episodes)

streamlit run app.py
```

Fresh clone with no weights? The home page and RL console offer one-click **fast training** buttons.

## 🧪 Architecture

```
Project-Lazarus/
├── app.py                    # command center (Streamlit)
├── pages/                    # the twelve consoles
├── lazarus/
│   ├── data/
│   │   ├── synthetic.py      # Briggs-model aDNA read simulator + presets
│   │   ├── species_db.py     # curated candidate dossiers + simulated gene templates
│   │   ├── importers.py      # CSV/TSV/FASTA/FASTQ/JSON/PDF → ImportResult{reads, notes, meta}
│   │   └── lims.py           # SQLite specimen registry + chain-of-custody (#106, #107)
│   ├── core/
│   │   ├── adna.py           # damage profiles, δ/λ fitting, contamination, verdicts
│   │   ├── phylogeny.py      # k-mer/Mash distances, neighbor joining, MDS
│   │   ├── crispr.py         # codon edit extraction, guide + pegRNA design
│   │   ├── qc.py             # #8–#13  QC, PMD, complexity, dedup, trimming, UDG
│   │   ├── assembly.py       # #20–#23 misregistration, consensus, coverage, k-mer spectra
│   │   ├── evolution.py      # #33–#34 Newick surgery + JC69/K2P/HKY85 models
│   │   ├── wetlab.py         # #67, #68, #88–#90 ssODN, cascade, routes, surrogates, ethics
│   │   ├── ops.py            # #110, #112, #113 lockers, benchmark rota, reports
│   │   └── funcgen.py        # ORFs, translation, codon usage, protein biophysics
│   ├── ml/
│   │   ├── features.py       # 24-dim fragmentomic feature vectors, profile tensors
│   │   ├── models.py         # ReadAuthenticityMLP, DamageProfileCNN (PyTorch)
│   │   ├── train.py          # training entry point → models/
│   │   └── inference.py      # lazy façade with heuristic fallbacks
│   ├── rl/
│   │   ├── genome_env.py     # GenomeGapFill-v0 + baseline policies
│   │   ├── agent.py          # DQN (replay, target net, ε-greedy)
│   │   └── train.py          # training + benchmark → models/
│   ├── registry.py           # 115-tool registry + tool_by_id() helper
│   └── viz/plots.py          # dark-theme plotly figure factory
├── models/                   # trained artefacts (committed)
├── scripts/app_sweep.py      # pre-push gate: compile + AppTest every Streamlit page
└── tests/                    # pytest suite (36 tests)
```

## ✅ Verifying a change

```bash
python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest tests/ -q          # 36 passed
.venv/bin/python scripts/app_sweep.py         # 13/13 clean
```

`scripts/app_sweep.py` does two independent checks per file, because `AppTest.from_file`
alone is not a sufficient gate: a page containing a `SyntaxError` prints a traceback but
leaves `AppTest.exception` empty, so it would otherwise false-pass. The sweep therefore
`py_compile`s every entry point first, then runs it through `AppTest` and judges
`at.exception`.

## 🧠 The ML/RL in one paragraph

Ancient libraries are short, terminally deaminated and contaminated with modern DNA — so **read-level authenticity** is a classification problem (24 fragmentomic features → MLP) and **library damage tier** is a curve-classification problem (2×20 misincorporation curves → 1-D CNN). Genome reconstruction is sequential decision-making under three noisy sources of truth — a
damaged-library read pileup, its quality curve and an extant-relative genome that is
**misregistered** wherever lineage-specific indels shift it — which is exactly the MDP
`GenomeGapFill-v0` models: state = local assembly window + pileup call + quality + reference +
gap mask + lag-agreement registration statistics, action = base call, reward = correctness. A DQN
learns to re-register the reference on the fly and weigh reads against it, benchmarked against
reference-greedy, pileup-greedy, quality-aware, random and a hand-crafted registration rule.
On the shipped evaluation the DQN reaches **0.741** mean gap-fill accuracy against the
registration rule's **0.783** — i.e. **competitive with** the rule baseline, not superior to it;
both are far ahead of the naive heuristics.

Training and benchmarking are versioned: **#110** retrains into `models/locker_<tag>/`
(weights + evaluation JSON + env hyper-parameters + wall-clock), **#112** runs the nightshift
rota and the benchmark suite against the shipped checkpoint, and **#113** renders the repo
status wall and exports JSON/CSV/HTML reports.

## 📜 Data honesty

* All read sets, gene templates and lineage sequences are **simulated** (explicitly labelled in the UI). Gene templates are *inspired by* real stories — e.g. mammoth cold-adapted hemoglobin (Campbell et al., 2010) and the Christmas Island rat genome recovery (van der Valk et al., 2022) — but are **not** validated extinct alleles.
* Species dossiers follow the public record (extinction dates, relatives, revival routes); factor scores are illustrative engineering estimates.
* Guide-RNA and off-target scoring are simplified heuristics, not Doench/CFD models. Not a wet-lab protocol.
* QC thresholds, library-complexity fits, k-mer genome-size estimates and substitution-model distances are textbook-style estimators on simulated data — illustrative, not measurements.
* Revival routes, surrogate rankings and the ethics checklist are decision-support aids, not a costed programme, a veterinary opinion, or an ethics approval.
* The LIMS is a local SQLite file (`lims.sqlite`, git-ignored) for demo provenance tracking.

## ⚖️ Ethics

Real de-extinction touches animal welfare (elephant surrogacy), biosafety, conservation opportunity cost and the law of the land. Lazarus is built for education, tooling research and responsible debate — every revival route in the scorecard should be read through that lens.

## 📖 References in the spirit of the work

- Briggs et al. (2007) patterns of damage in ancient DNA
- Campbell et al. (2010) mammoth hemoglobin cold adaptation
- Jónsson et al. mapDamage; Green et al. PMDtools-style damage-aware authentication
- Ondov et al. (2016) Mash; Saitou & Nei (1987) neighbor joining
- van der Valk et al. (2022) functional genomics of the extinct Christmas Island rat
- Jukes & Cantor (1969) · Kimura (1980) 2-parameter · Hasegawa, Kishino & Yano (1985)
- Ran et al. prime editing; Skov et al. PMDtools posterior damage scores
- Mnih et al. (2015) DQN · Brockman et al. Gymnasium

---

*Apache-2.0 · Project Lazarus — because extinction should be a compile error, not a runtime one.*
