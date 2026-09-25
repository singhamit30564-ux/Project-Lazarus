"""📥 Import Studio — CSV / PDF / FASTA / FASTQ / JSON / TSV ingest (tools #105–#107)."""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import streamlit as st

from lazarus import ui_common as ui
from lazarus.core import adna
from lazarus.data import lims
from lazarus.data.importers import (
    ImportResult, import_bytes, import_file, import_text,
)
from lazarus.data.synthetic import PRESETS, simulate_read_set
from lazarus.titan.ui import dr_titan
from lazarus.viz import plots

st.set_page_config(page_title="Import Studio · Lazarus", page_icon="📥", layout="wide")
ui.inject_css()
ui.sidebar_brand()

st.title("📥 Import Studio")
st.caption(
    "Drop any sequence file — CSV, TSV, FASTA, FASTQ, JSON or a PDF methods dump — and Lazarus "
    "turns it into authenticated reads, a donor/target edit pair, or a registered specimen with "
    "a chain-of-custody ledger. Tools #105 (format converter), #106 (specimen LIMS) and "
    "#107 (chain-of-custody tracker) live here."
)

dr_titan("import", note="Tools #105–107 · Division X")

ACCEPTED = ["csv", "tsv", "txt", "fasta", "fa", "fna", "fq", "fastq", "json", "pdf"]


# ---------------------------------------------------------------------------
# Bundled sample fixtures (so the console is never empty)
# ---------------------------------------------------------------------------

def bundled_samples() -> dict[str, tuple[bytes, str]]:
    from lazarus.data.species_db import GENE_TEMPLATES

    tpl = GENE_TEMPLATES["mammoth_hbb"]
    rs = simulate_read_set(n_reads=180, frag_mean=46, damage_5p=0.30, damage_3p=0.22,
                           contamination=0.10, species="Mammuthus primigenius", seed=17)
    fasta = "".join(f">read_{i:04d} mammoth molar\n{r.seq}\n" for i, r in enumerate(rs.reads[:120]))
    fastq = "".join(f"@read_{i:04d}\n{r.seq}\n+\n{'I' * len(r.seq)}\n"
                    for i, r in enumerate(rs.reads[:60]))
    csv_reads = "read_id,sequence,qual\n" + "".join(
        f"r{i},{r.seq},35\n" for i, r in enumerate(rs.reads[:80]))
    csv_pair = ("gene,donor_cds,target_cds\n"
                f"{tpl.gene_name},{tpl.donor_dna},{tpl.target_dna}\n")
    fasta_pair = (f">donor {tpl.donor_species}\n{tpl.donor_dna}\n"
                  f">target ancestral allele\n{tpl.target_dna}\n")
    return {
        "FASTA · simulated mammoth reads": (fasta.encode(), "mammoth_reads.fasta"),
        "FASTQ · simulated mammoth reads": (fastq.encode(), "mammoth_reads.fastq"),
        "CSV · one read per row": (csv_reads.encode(), "mammoth_reads.csv"),
        "CSV · paired donor / target CDS": (csv_pair.encode(), "hbb_pair.csv"),
        "FASTA · paired donor / target CDS": (fasta_pair.encode(), "hbb_pair.fasta"),
    }


with st.sidebar:
    ui.section("Input source")
    source_mode = st.radio(
        "Choose", ["Upload a file", "Bundled sample", "Paste sequences"],
        label_visibility="collapsed",
    )
    st.caption(f"Accepted: {', '.join('.' + a for a in ACCEPTED)}")

result: ImportResult | None = None
source_label = "—"

if source_mode == "Upload a file":
    up = st.file_uploader("Sequence file", type=ACCEPTED, accept_multiple_files=False)
    if up is not None:
        try:
            result = import_file(up, up.name)
            source_label = up.name
        except Exception as exc:
            st.error(f"Import failed: {type(exc).__name__}: {exc}")
elif source_mode == "Bundled sample":
    samples = bundled_samples()
    pick = st.selectbox("Sample", list(samples))
    data, fname = samples[pick]
    result = import_bytes(data, fname)
    source_label = fname
    st.caption("Bundled fixtures are SIMULATED — they exist to exercise the importer.")
else:
    pasted = st.text_area(
        "Paste FASTA / FASTQ / bare sequences (or a PDF text dump)",
        height=200, placeholder=">read1\nACGTACGT…\n>read2\nACGTNNNN…",
    )
    if st.button("🔬 Import pasted text", width='stretch'):
        res = import_text(pasted or "", "pasted")
        if res.n:
            result, source_label = res, "pasted text"
            st.session_state["import_result"] = res
    if result is None and "import_result" in st.session_state:
        result = st.session_state["import_result"]
        source_label = "pasted text"

if result is None:
    st.info("Upload a file, pick a bundled sample, or paste sequences to begin.")
    ui.footer()
    st.stop()

if not result.reads:
    st.warning("The file was read, but no usable sequence came out of it.")
    for n in result.notes:
        st.caption(f"· {n}")
    ui.footer()
    st.stop()

# ---------------------------------------------------------------------------
# Import summary
# ---------------------------------------------------------------------------
summary = result.summary()
ui.section("Import summary", result.meta.get("format", "—"))
st.markdown(
    f"<div class='lz-card'><span class='lz-tag'>{source_label}</span> "
    f"<span class='lz-tag warn'>{summary['reads']} reads</span> "
    f"<span class='lz-tag mute'>{summary['total_bases']:,} bp</span> "
    f"<span class='lz-tag {'warn' if result.paired else 'mute'}'>"
    f"{'paired donor/target' if result.paired else 'read set'}</span></div>",
    unsafe_allow_html=True,
)
for n in result.notes:
    st.caption(f"· {n}")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Reads", summary["reads"])
k2.metric("Mean length", f"{summary['mean_length']:.0f} bp")
k3.metric("Median length", f"{summary['median_length']:.0f} bp")
k4.metric("Total bases", f"{summary['total_bases']:,}")

reads = result.reads
lengths = np.array([len(r.seq) for r in reads], dtype=float)

tab_preview, tab_auth, tab_pair, tab_convert, tab_lims = st.tabs([
    "📊 Preview", "🧬 Authenticate", "✂️ Donor / target pair",
    "🔄 #105 Format converter", "🗃️ #106 + #107 LIMS & custody",
])

# --------------------------------------------------------------------------- preview
with tab_preview:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.plotly_chart(plots.fig_fragment_lengths(lengths), width='stretch')
    with c2:
        st.markdown("**First 25 sequences**")
        st.dataframe(
            pd.DataFrame({
                "#": list(range(1, min(len(reads), 25) + 1)),
                "length": [len(r.seq) for r in reads[:25]],
                "sequence": [r.seq[:46] + ("…" if len(r.seq) > 46 else "") for r in reads[:25]],
            }),
            hide_index=True, width='stretch', height=420,
        )
    st.caption(
        "Sequences are shown as imported. Nothing has been aligned, de-duplicated or "
        "authenticated yet — do that in the Authenticate tab or the QC console."
    )

# --------------------------------------------------------------------------- auth
with tab_auth:
    if st.button("🧬 Run authentication on this import", width='stretch'):
        with st.spinner("Profiling damage, fragmentomics and contamination …"):
            st.session_state["import_auth"] = adna.analyze(reads)
    auth = st.session_state.get("import_auth")
    if auth is None:
        st.info("Run the authentication bundle to score this library.")
    else:
        a1, a2 = st.columns([1, 1.4])
        with a1:
            st.plotly_chart(plots.fig_gauge(auth["authenticity_score"]), width='stretch',
                            config={"displayModeBar": False})
            st.markdown(auth["verdict"], unsafe_allow_html=True)
        with a2:
            st.plotly_chart(plots.fig_misincorporation(auth["profile"]), width='stretch')
        dp = auth["damage_params"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("δS", f"{dp['delta_s']:.3f}")
        m2.metric("δD", f"{dp['delta_d']:.3f}")
        m3.metric("λ (bp)", f"{dp['lambda']:.1f}")
        m4.metric("Contamination (heuristic)", f"{100 * auth['contamination_heuristic']:.1f}%")

# --------------------------------------------------------------------------- pair
with tab_pair:
    if not result.paired:
        st.info(
            "This import is a plain read set. Upload a **paired** file (two equal-length "
            "codon-aligned sequences, or CSV columns named `donor_cds` / `target_cds`) to "
            "design codon edits against it."
        )
    else:
        donor = result.meta.get("donor_sequence", "")
        target = result.meta.get("target_sequence", "")
        st.success(
            f"Paired import detected — donor `{result.meta.get('donor_name', '?')}` ↔ "
            f"target `{result.meta.get('target_name', '?')}`. These two records feed the "
            "CRISPR Edit Planner (tools #63–68)."
        )
        c1, c2 = st.columns(2)
        c1.metric("Donor length", f"{len(donor)} nt")
        c2.metric("Target length", f"{len(target)} nt")
        if len(donor) == len(target) and len(donor) % 3 == 0:
            from lazarus.core.crispr import extract_codon_edits

            edits = extract_codon_edits(donor.upper(), target.upper())
            st.markdown(f"**{len(edits)} codon edit(s)** required to convert donor → target.")
            if edits:
                st.dataframe(pd.DataFrame([{
                    "codon #": e.codon_index + 1,
                    "aa": f"{e.from_aa}→{e.to_aa}",
                    "codon": f"{e.from_codon}→{e.to_codon}",
                    "nt edits": " ".join(f"{a}{p + 1}{b}" for p, a, b in e.nt_changes),
                } for e in edits]), hide_index=True, width='stretch', height=300)
                st.caption("Open the ✂️ CRISPR Edit Planner and paste these as a custom pair "
                           "to design guides, ssODN donors and a multiplex cascade.")
        else:
            st.warning(
                f"Lengths differ ({len(donor)} vs {len(target)} nt) or are not a multiple of 3 — "
                "codon-level edit planning needs equal-length, codon-aligned CDS pairs."
            )
        with st.expander("Raw paired sequences", expanded=False):
            st.code(f">donor\n{donor}", language="text")
            st.code(f">target\n{target}", language="text")

# --------------------------------------------------------------------------- converter
with tab_convert:
    ui.section("Sequence Format Converter", "tool #105")
    fmt = st.selectbox("Export as", ["FASTA", "FASTQ (Q30 flat)", "CSV", "JSON", "plain text"])
    line_wrap = st.slider("FASTA line width", 40, 120, 60, 10)

    def to_fasta(seqs: list[tuple[str, str]]) -> str:
        out = []
        for name, s in seqs:
            out.append(f">{name}")
            for i in range(0, len(s), line_wrap):
                out.append(s[i: i + line_wrap])
        return "\n".join(out) + "\n"

    names: list[str] = result.meta.get("labels") or [f"seq_{i + 1}" for i in range(len(reads))]
    pairs = [(names[i] if i < len(names) else f"seq_{i + 1}", r.seq)
             for i, r in enumerate(reads)]

    if fmt == "FASTA":
        payload, mime, ext = to_fasta(pairs), "text/plain", "fasta"
    elif fmt.startswith("FASTQ"):
        payload = "".join(f"@{n}\n{s}\n+\n{'I' * len(s)}\n" for n, s in pairs)
        mime, ext = "text/plain", "fastq"
    elif fmt == "CSV":
        buf = io.StringIO()
        pd.DataFrame({"read_id": [n for n, _ in pairs],
                      "length": [len(s) for _, s in pairs],
                      "sequence": [s for _, s in pairs]}).to_csv(buf, index=False)
        payload, mime, ext = buf.getvalue(), "text/csv", "csv"
    elif fmt == "JSON":
        payload = json.dumps({"source": source_label,
                              "format": result.meta.get("format"),
                              "n_sequences": len(pairs),
                              "sequences": [{"id": n, "sequence": s} for n, s in pairs]},
                             indent=2)
        mime, ext = "application/json", "json"
    else:
        payload = "\n".join(s for _, s in pairs) + "\n"
        mime, ext = "text/plain", "txt"

    st.code(payload[:1200] + ("\n…" if len(payload) > 1200 else ""), language="text")
    st.download_button(f"⬇️ Download {fmt}", payload,
                       file_name=f"lazarus_converted.{ext}", mime=mime, width='stretch')

# --------------------------------------------------------------------------- lims
with tab_lims:
    ui.section("Specimen LIMS", "tool #106")
    lims.init_db()
    with st.form("register"):
        c1, c2, c3 = st.columns(3)
        sid = c1.text_input("Specimen ID", value=f"LZ-{abs(hash(source_label)) % 900 + 100:03d}")
        taxon = c2.text_input("Taxon", value=result.meta.get("species", "Mammuthus primigenius"))
        origin = c3.text_input("Origin / locality", value="Wrangel Island")
        c4, c5, c6 = st.columns(3)
        collector = c4.text_input("Collected by", value="Lazarus field team")
        collected_on = c5.text_input("Collected on", value="2026-09-25")
        permit = c6.text_input("Permit / accession", value="—")
        notes_in = st.text_input("Notes", value="; ".join(result.notes[:2])[:180])
        submitted = st.form_submit_button("🗃️ Register specimen + log upload", width='stretch')

    if submitted:
        spec = lims.register_specimen(
            specimen_id=sid, taxon=taxon, common_name="", origin=origin,
            collected_by=collector, collected_on=collected_on, permit=permit,
            source_format=result.meta.get("format", ""), n_reads=len(reads), notes=notes_in,
        )
        lims.register_library(lib_id=f"{sid}-L1", specimen_id=sid, n_reads=len(reads),
                              mean_len=float(np.mean(lengths)) if lengths.size else 0.0)
        st.success(f"Registered `{sid}` and logged the upload event.")
        st.rerun()

    st.markdown("**Registered specimens**")
    rows = lims.list_specimens()
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width='stretch', height=260)
    else:
        st.info("No specimens registered yet.")

    st.divider()
    ui.section("Chain of Custody Tracker", "tool #107")
    if rows:
        ids = [r["id"] for r in rows]
        chosen = st.selectbox("Specimen", ids, index=0)
        with st.form("event"):
            e1, e2, e3 = st.columns(3)
            action = e1.selectbox("Action", list(lims.CUSTODY_ACTIONS))
            actor = e2.text_input("Actor", value="Lazarus Import Studio")
            location = e3.text_input("Location", value="clean room")
            detail = st.text_input("Detail", value="handled under LIMS protocol")
            add = st.form_submit_button("⛓️ Append custody event", width='stretch')
        if add:
            lims.log_event(chosen, action, actor=actor, location=location, detail=detail)
            st.rerun()

        chain = lims.chain_of_custody(chosen)
        if chain:
            st.dataframe(pd.DataFrame(chain), hide_index=True, width='stretch', height=280)
            integ = lims.custody_integrity(chosen)
            if integ["ok"]:
                st.success(f"Chain intact — {integ['n_events']} event(s), "
                           f"{integ['first']} → {integ['last']}.")
            else:
                st.warning("Integrity flags: " + "; ".join(integ["problems"]))
            st.download_button("⬇️ Export LIMS (JSON)", lims.export_json(),
                               file_name="lazarus_lims.json", mime="application/json",
                               width='stretch')
        else:
            st.info("No custody events yet for this specimen.")
    else:
        st.info("Register a specimen above to start its custody chain.")

st.caption(
    "Imported sequences are treated as untrusted input: they are parsed, never executed, and "
    "everything downstream (damage profiles, edit plans) is a design heuristic on simulated data."
)
ui.footer()
