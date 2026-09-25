"""Dr. Titan's field notes — curated expert tips per console.

Persona: Dr. Titan, Lazarus chief science advisor. A paleogenomics polymath who
has buried more libraries than most people have sequenced. Tips are practical,
scientific, and occasionally sharp-edged.
"""

PERSONA = {
    "name": "Dr. Titan",
    "title": "Chief Science Advisor, Project Lazarus",
    "motto": "Extinction is a data problem until it isn't.",
}

TIPS: dict[str, list[str]] = {
    "home": [
        "Start with authentication. A perfect assembly of contaminated DNA is a beautifully organized landfill.",
        "Run the Ancient DNA Analyzer on any library BEFORE phylogenetic placement — damage signatures tell you when the tree is about to lie to you.",
        "De-extinction is 20% molecular biology and 80% logistics. The scorecard's repro-tractability column is where dreams go to quietly die.",
        "Everything here is a rehearsal. Simulated data first, permafrost later — ego and ancient DNA don't mix.",
        "100+ tools, one rule: never trust a reference genome you didn't audit for misregistration first.",
    ],
    "adna": [
        "C→T at 5′, G→A at 3′ — that pattern is your authentication. If it also appears in the interior, you have heat damage, UV damage, or a bug.",
        "A median fragment length under 50 bp isn't a defect, it's a fingerprint. Fight the urge to size-select your data into looking modern.",
        "Contamination is measured against expectations: fresh elephant tissue must never authenticate as ancient. Always run the negative control.",
        "δS without δD is half a story. Single-strand overhangs decay (that's your λ), double-strand lesions persist for millennia.",
        "Reference-free composition proxy = smoke detector, not mass spectrometer. Align before you publish.",
        "Watch the ML contamination estimate drift away from the heuristic — when they disagree loudly, one of them found something real.",
    ],
    "phylo": [
        "k-mer distance first, likelihood later. If the k-mer tree is wrong, no bootstrap on Earth will save you.",
        "7-mers work for close mammals. For 200 bp aDNA fragments and deep splits, shrink k — or better, place the query on a fixed backbone instead of rebuilding the tree.",
        "An unrooted tree has no ancestors, only relatives. Rooting is a hypothesis — draw it dashed.",
        "Your query is degraded and the references are pristine. Real placement tools down-weight damaged sites; watch yours not to over-cluster with whatever is AT-rich.",
        "If the MDS plot and the cladogram disagree, the distance matrix is telling you it doesn't trust its own geometry. Listen.",
    ],
    "rl": [
        "A relative genome is a liar with good intentions: lineage-specific indels shift coordinates and quietly corrupt every 'obvious' base. That's the whole game here.",
        "Watch the lag-agreement profile. When all five lags go flat, no source knows the truth — that's where policy quality actually shows.",
        "A hand rule beating your agent is free tuition: steal its logic, turn it into features, train again. That's not cheating, that's science.",
        "Reward +1/−0.5 keeps the agent honest about rare bases. Plain accuracy hides the pain of G/C errors.",
        "If gap-fill accuracy ever exceeds 1 − divergence on reference-greedy worlds, check for leakage before you celebrate.",
    ],
    "crispr": [
        "Design the protein first, the codons second, the guides last. A perfect guide to the wrong codon is a scalpel in the dark.",
        "A missing PAM is not a dead end — it's a prime-editing invitation. Keep PE pegRNAs in your back pocket.",
        "Off-target scanning on a 300 bp gene is a demo. Before ordering anything: genome-wide off-targets, two independent clones, whole-genome validation.",
        "ssODN donors want 40+ bp homology arms and phosphorothioate ends. The cell repairs DNA; it doesn't do windows.",
        "One edit is a proof of concept. Three edits is a program. Ten edits is a regulatory dossier — start writing it early.",
        "Resurrecting ancestral alleles is the easy part. Expressing them in the right cell type, at the right dosage, in the right trimester — that's engineering.",
    ],
    "scorecard": [
        "Weights are arguments in disguise. Slide ecosystem-fit to 1.0 and watch which projects survive the thought experiment.",
        "The bucardo proves the pipeline runs. Its clone dying after ten minutes proves the pipeline has a soul — lung defects teach humility.",
        "Every candidate is somebody's cause: the ethics score isn't politeness, it's risk-weighting with a conscience.",
        "If reproductive tractability sits under 0.3, budget a decade and a fallback species. Looking at you, Steller's sea cow.",
        "Extinction dates matter for data: 1936 thylacine tissue is chemistry; 2000 BC mammoth permafrost is a lottery ticket. Rank accordingly.",
    ],
    "palette": [
        "100+ tools, one question each. If you can't decide which to open, open the Authenticator — it never lies about your data's age.",
        "Tools marked LIVE are sharp. Tools marked WAVE are under construction. Tools marked ROADMAP are promises — mind your fingers.",
        "Every big discovery here is three boring tools chained: parse → quantify → judge. Skip the middle and the judge hallucinates.",
    ],
    "funcgen": [
        "An ORF without a Kozak context is a rumor with a start codon.",
        "Codon optimization is host flattery: a mammoth protein in elephant cells must speak elephant — not E. coli.",
        "GRAVY and pI are horoscopes for proteins. Entertaining, occasionally right, never sufficient.",
        "Hydropathy peaks earn their TM-helix call at ≥19 residues. Below that, it's probably just a sticky patch.",
        "Motif hits are hypotheses with grammar. 'H…D…S' is a catalytic triad candidate, not an enzyme.",
    ],
    "protein": [
        "Ancestral proteins are not fossils — they're the consensus of everything that survived. Reconstruct, then verify thermostability experimentally.",
        "Mammoth hemoglobin taught us two substitutions can move an oxygen curve. Count mutations in phenotypes, not in nucleotides.",
        "An interface that flips four times across a phylogeny is either under selection or badly aligned. Check alignment first, always.",
    ],
    "synth": [
        "Mask your repeats before designing anything. Transposons are where assemblies go to confess.",
        "Telomeres and centromeres are the parts of 'complete genome' that quietly weren't. Track them like debt.",
        "A synthetic chromosome is a shipping container: plan the cargo (genes), the locks (safeguards), and the manifest (provenance) before the voyage.",
    ],
    "import": [
        "Provenance beats throughput: a scrappy 200-read library with a clean chain of custody is worth more than a terabyte with a shrug.",
        "CSV columns lie quietly. Always eyeball the first row before you let a parser decide which column is the sequence.",
        "A PDF methods section is a sequence mine, not a sequence database. Treat every extracted run as a lead to verify, not a result.",
        "Paired donor/target uploads are the CRISPR planner's favourite meal: two equal-length codon-aligned CDS records and everything downstream works.",
        "Log the upload into the LIMS the moment it lands. Six months from now nobody remembers which 'final_v3.fasta' was the real one.",
    ],
    "qc": [
        "FastQC was written for modern libraries. On aDNA, a 'failed' per-base quality module is often just deamination doing its job.",
        "Complexity, not read count, is your real budget. A saturating rarefaction curve means more sequencing buys duplicates.",
        "Duplicate removal on aDNA is a knife, not a broom: genuine short fragments collide by chance, and you will throw away real molecules.",
        "Partial UDG is the default for a reason — it keeps the terminal two bases you authenticate with and cleans the interior you genotype from.",
        "A PMD score near 0.5 is not a coin flip, it's a short read. Posterior confidence scales with the number of informative terminal positions.",
        "Trim for biology, not for beauty: over-trimming a 40 bp fragment to 'improve' Q30 destroys the damage signal you came for.",
    ],
    "assembly": [
        "Assembly is where reference bias puts on a lab coat and starts giving orders. Audit the registration before you trust a single base.",
        "The lag-agreement profile is the whole trick: if the best lag changes across the contig, an indel moved and every downstream coordinate is fiction.",
        "Depth averages hide dropout. Breadth at ≥5× is the number that decides whether a genotype call is defensible.",
        "Consensus callers disagree exactly where the data is damaged — that disagreement is a feature map of your weak sites.",
        "A k-mer spectrum with a towering 1× spike is sequencing error; a fat right shoulder is repeats. Learn to tell them apart before quoting a genome size.",
        "Never let a pileup call outvote physics: at low coverage the reference wins by default, and the reference is a relative, not the truth.",
    ],
    "evolution": [
        "Rerooting is a hypothesis with a coordinate system. Do it in a copy, and write down who the outgroup was.",
        "Uncorrected p-distance lies after about 15% divergence. Jukes-Cantor is the minimum courtesy; K2P if transitions outnumber transversions.",
        "A κ far above 2 means either real transition bias or an alignment that has stopped trying. Check the alignment first.",
        "When every correction blows up to NaN, you are saturated — no amount of mathematics recovers substitutions that already overwrote each other.",
        "Newick is a file format and a liability: one unlabelled polytomy can quietly reorder your entire figure legend.",
        "Tree height is not time. Convert to dates only with a clock model and a calibration you are willing to defend in public.",
    ],
    "ops": [
        "Revival programs fail in the spreadsheet long before the lab: schedule, surrogates, regulatory runway — in that order of severity.",
        "Genetic load is compounding interest on extinction. Pay it down with adaptive introgression or pay it later with failed gestations.",
        "Ethics review isn't a gate, it's an instrument panel. If you can't read it, don't fly.",
        "Retrain into a locker, never onto the shipped weights. A benchmark you cannot reproduce from a checkpoint is a horoscope.",
        "Ship the checkpoint with the claim: accuracy without a versioned artefact is a rumour with a decimal point.",
        "The nightshift is for jobs that must not surprise anyone: sweeps, benchmarks, integrity checks — run them locked, log everything.",
        "A status wall is not vanity. If you can't see the repo's health at a glance, you'll debug it at 3 a.m. instead.",
    ],
    "cryo": [
        "Celia's clone was born alive and died in minutes. SCNT success is a distribution with a cruel left tail — model it that way.",
        "iPSC lines drift. Bank early, passage less, karyotype always.",
        "An artificial womb is a thermostat with lawyers. Parameter sheets save years.",
    ],
    "dataops": [
        "Provenance or it didn't happen: chain-of-custody beats cleverness when a courtroom asks where the mammoth came from.",
        "Metadata is the DNA of data. Half-annotated FASTA files are the subfossils of bioinformatics.",
        "Benchmarks without versioned weights are horoscopes. Ship the checkpoint with the claim.",
    ],
    "generic": [
        "Question your reference before you question your results.",
        "Controls first, discovery second. The negative control is the only colleague who never flatters you.",
        "If a result can't survive being re-derived from raw reads, it was never a result — it was a mood.",
    ],
}


def tips_for(key: str) -> list[str]:
    return TIPS.get(key) or TIPS["generic"]
