# 🦴 Project Lazarus

**De-Extinction Intelligence Suite** — a Streamlit + ML/RL workbench for bringing the dead back to life (digitally first).

Authenticate ancient DNA. Place a specimen on the tree of life. Reconstruct its shattered genome with a reinforcement-learning agent. Plan the CRISPR edits that re-wire a living relative toward the lost ancestor. Rank which species should walk the Earth again.

---

## ✨ The five consoles

| Console | What it does | Engine |
|---------|--------------|--------|
| 🧬 **Ancient DNA Analyzer** | mapDamage-style terminal misincorporation profiles (C→T at 5′, G→A at 3′), Briggs damage-parameter fitting (δ<sub>S</sub>, δ<sub>D</sub>, λ), fragment-length fragmentomics, contamination estimate, 0–100 authenticity verdict | damage chemistry model + read-authenticity MLP + damage-profile CNN |
| 🌳 **Phylogenetic Placement** | Mash-style k-mer distances → neighbor-joining cladogram → classical MDS embedding of an ancient query among candidate relatives | k-mer statistics + NJ |
| 🤖 **Genome Reconstruction RL** | `GenomeGapFill-v0` Gymnasium env: the agent fills every gap of a degraded genome by re-registering a misindel-shifted relative reference against noisy read pileups | DQN vs registration-rule / reference-greedy / pileup-greedy / quality-aware / random baselines |
| ✂️ **CRISPR Edit Planner** | codon-level edit table that resurrects ancestral proteins in the relative's genome; SpCas9 guide design with off-target seed heuristics; prime-editing pegRNA fallback | guide scoring heuristics |
| 🦣 **Candidate Scorecard** | weighted multi-criteria feasibility ranking of real candidates (mammoth, thylacine, dodo, great auk, bucardo, Steller's sea cow, Christmas Island rat, Irish elk) | decision model |

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
├── pages/                    # the five consoles
├── lazarus/
│   ├── data/
│   │   ├── synthetic.py      # Briggs-model aDNA read simulator + presets
│   │   └── species_db.py     # curated candidate dossiers + simulated gene templates
│   ├── core/
│   │   ├── adna.py           # damage profiles, δ/λ fitting, contamination, verdicts
│   │   ├── phylogeny.py      # k-mer/Mash distances, neighbor joining, MDS
│   │   └── crispr.py         # codon edit extraction, guide + pegRNA design
│   ├── ml/
│   │   ├── features.py       # 24-dim fragmentomic feature vectors, profile tensors
│   │   ├── models.py         # ReadAuthenticityMLP, DamageProfileCNN (PyTorch)
│   │   ├── train.py          # training entry point → models/
│   │   └── inference.py      # lazy façade with heuristic fallbacks
│   ├── rl/
│   │   ├── genome_env.py     # GenomeGapFill-v0 + baseline policies
│   │   ├── agent.py          # DQN (replay, target net, ε-greedy)
│   │   └── train.py          # training + benchmark → models/
│   └── viz/plots.py          # dark-theme plotly figure factory
├── models/                   # trained artefacts (committed)
└── tests/                    # pytest suite
```

## 🧠 The ML/RL in one paragraph

Ancient libraries are short, terminally deaminated and contaminated with modern DNA — so **read-level authenticity** is a classification problem (24 fragmentomic features → MLP) and **library damage tier** is a curve-classification problem (2×20 misincorporation curves → 1-D CNN). Genome reconstruction is sequential decision-making under three noisy sources of truth — a
damaged-library read pileup, its quality curve and an extant-relative genome that is
**misregistered** wherever lineage-specific indels shift it — which is exactly the MDP
`GenomeGapFill-v0` models: state = local assembly window + pileup call + quality + reference +
gap mask + lag-agreement registration statistics, action = base call, reward = correctness. A DQN
learns to re-register the reference on the fly and weigh reads against it, benchmarked against
reference-greedy, pileup-greedy, quality-aware, random and a hand-crafted registration rule.

## 📜 Data honesty

* All read sets, gene templates and lineage sequences are **simulated** (explicitly labelled in the UI). Gene templates are *inspired by* real stories — e.g. mammoth cold-adapted hemoglobin (Campbell et al., 2010) and the Christmas Island rat genome recovery (van der Valk et al., 2022) — but are **not** validated extinct alleles.
* Species dossiers follow the public record (extinction dates, relatives, revival routes); factor scores are illustrative engineering estimates.
* Guide-RNA and off-target scoring are simplified heuristics, not Doench/CFD models. Not a wet-lab protocol.

## ⚖️ Ethics

Real de-extinction touches animal welfare (elephant surrogacy), biosafety, conservation opportunity cost and the law of the land. Lazarus is built for education, tooling research and responsible debate — every revival route in the scorecard should be read through that lens.

## 📖 References in the spirit of the work

- Briggs et al. (2007) patterns of damage in ancient DNA
- Campbell et al. (2010) mammoth hemoglobin cold adaptation
- Jónsson et al. mapDamage; Green et al. PMDtools-style damage-aware authentication
- Ondov et al. (2016) Mash; Saitou & Nei (1987) neighbor joining
- van der Valk et al. (2022) functional genomics of the extinct Christmas Island rat
- Mnih et al. (2015) DQN · Brockman et al. Gymnasium

---

*Apache-2.0 · Project Lazarus — because extinction should be a compile error, not a runtime one.*
