#!/usr/bin/env python3
"""
SF Sorting Hat — Streamlit UI (terminal aesthetic, same quiz logic as sorting_hat.py).
Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import html
import random

import streamlit as st

from sorting_hat import (
    DATAMUSE_SEEDS,
    LOCAL_WORD_BANK,
    adaptive_sample,
    compute_vibe_match,
    fetch_extended_bank,
    vec_add,
)
from sorting_hat_pca import DEFAULT_HOUSES, _DIM_LABELS, _K, _PCA, project

# Mirrors sorting_hat.py ANSI palette (web)
C = {
    "red": "#ff5555",
    "yellow": "#f1fa8c",
    "green": "#50fa7b",
    "cyan": "#8be9fd",
    "blue": "#6272a4",
    "pink": "#ff79c6",
    "fg": "#f8f8f2",
    "dim": "#6e6e6e",
}

ANSI_TO_HEX = {
    "\033[91m": C["red"],
    "\033[93m": C["yellow"],
    "\033[92m": C["green"],
    "\033[96m": C["cyan"],
    "\033[94m": C["blue"],
    "\033[95m": C["pink"],
}


def _bw_bar_html(pct: int, width: int = 26) -> str:
    """Light █ / dim ░ on dark background (read as monochrome meter)."""
    f = min(width, max(0, round(pct / 100 * width)))
    filled = "█" * f
    empty = "░" * (width - f)
    return (
        f'<span style="color:#e8e8e8">{html.escape(filled)}</span>'
        f'<span style="color:#3d3d3d">{html.escape(empty)}</span>'
    )


def _pca_bar(var_share: float, width: int = 36) -> str:
    n = min(width, max(0, round(var_share * width)))
    return "█" * n + "·" * (width - n)


def _term_css() -> None:
    st.markdown(
        """
<style>
  .stApp {
    background: #0d0d0d;
    color: #f8f8f2;
  }
  .block-container {
    padding-top: 1.5rem !important;
    padding-bottom: 1rem !important;
  }
  [data-testid="stHeader"] { background: #0d0d0d; }
  [data-testid="stToolbar"] { display: none !important; }
  footer { visibility: hidden; }
  .terminal {
    font-family: ui-monospace, "SF Mono", "Cascadia Code", Menlo, monospace;
    font-size: 0.92rem;
    line-height: 1.5;
    color: #f8f8f2;
    max-width: 46rem;
    margin: 0 auto;
  }
  /* One logical terminal row = one block (never wrap mid-line) */
  .term-line {
    font-family: inherit !important;
    font-size: inherit !important;
    margin: 0;
    padding: 0;
    white-space: pre !important;
    overflow-x: auto;
    word-break: normal;
    line-height: 1.45;
  }
  .term-header {
    text-align: center;
    margin: 0.5rem 0 1rem 0;
  }
  .term-header-inner {
    display: inline-block;
    text-align: left;
  }
  .term-header-inner .term-line {
    display: block;
    width: max-content;
    max-width: 100%;
  }
  pre.term-pre {
    font-family: inherit !important;
    font-size: inherit !important;
    margin: 0;
    white-space: pre !important;
    overflow-x: auto;
    word-break: normal;
  }
  /* Full-width row highlight (quiz / menu) */
  div[data-testid="stButton"] { width: 100%; }
  div[data-testid="stButton"] > div {
    background: transparent !important;
    box-shadow: none !important;
  }
  div[data-testid="stButton"] button {
    background: transparent !important;
    background-image: none !important;
    color: #f8f8f2 !important;
    border: none !important;
    border-radius: 0 !important;
    box-shadow: none !important;
    width: 100% !important;
    min-height: 2.1rem !important;
    justify-content: flex-start !important;
    text-align: left !important;
    padding: 0.35rem 0.6rem !important;
    font-family: ui-monospace, "SF Mono", Menlo, monospace !important;
    font-size: 0.92rem !important;
  }
  div[data-testid="stButton"] button:hover {
  }
  div[data-testid="stButton"] button:focus-visible {
    box-shadow: inset 0 0 0 1px #6272a4 !important;
    outline: none !important;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _ensure_word_bank() -> None:
    if "proj" in st.session_state:
        return
    with st.spinner(""):
        bank, n_new = fetch_extended_bank(LOCAL_WORD_BANK, DATAMUSE_SEEDS)
        st.session_state.proj = {w: project(v, _PCA) for w, v in bank.items()}
        st.session_state.n_datamuse_new = n_new
        st.session_state.bank_total = len(bank)


def _reset_quiz() -> None:
    st.session_state.phase = "quiz"
    st.session_state.q = 1
    st.session_state.score_vector = [0.0] * _K
    st.session_state.used = set()
    st.session_state.nouns = None


def _build_question() -> None:
    proj = st.session_state.proj
    used = st.session_state.used
    score_vector = st.session_state.score_vector
    pool = {w: v for w, v in proj.items() if w not in used}
    if len(pool) < 5:
        pool = dict(proj)
        st.session_state.used = set()
    nouns, hint = adaptive_sample(score_vector, pool, DEFAULT_HOUSES)
    random.shuffle(nouns)
    st.session_state.nouns = nouns
    st.session_state.hint = hint


def _apply_pick(idx: int) -> None:
    nouns = st.session_state.nouns
    word, vec = nouns[idx]
    st.session_state.score_vector = vec_add(st.session_state.score_vector, vec)
    st.session_state.used.update(w for w, _ in nouns)
    q = st.session_state.q
    if q >= 10:
        st.session_state.phase = "results"
        st.session_state.ranked = compute_vibe_match(
            st.session_state.score_vector, DEFAULT_HOUSES
        )
    else:
        st.session_state.q = q + 1
        st.session_state.nouns = None


def main() -> None:
    st.set_page_config(
        page_title="SF Sorting Hat",
        page_icon="🎩",
        layout="centered",
        initial_sidebar_state="collapsed",
    )
    _term_css()

    if "phase" not in st.session_state:
        st.session_state.phase = "welcome"

    _ensure_word_bank()

    st.markdown('<div class="terminal">', unsafe_allow_html=True)

    if st.session_state.phase == "welcome":
        _render_welcome()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if st.session_state.phase == "quiz":
        if st.session_state.nouns is None:
            _build_question()
        _render_quiz()
        st.markdown("</div>", unsafe_allow_html=True)
        return

    _render_results()
    st.markdown("</div>", unsafe_allow_html=True)


def _render_welcome() -> None:
    n_new = st.session_state.get("n_datamuse_new", 0)
    total = st.session_state.get("bank_total", 0)

    st.markdown(
        "SF Sorting Hat"
    )

    k = _PCA["k"]
    cum = _PCA["cumvar"]
    pca_rows = [
        f"  ── {k} personality axes derived by PCA ({cum:.0%} variance explained) ──",
    ]
    for i, (lbl, v) in enumerate(zip(_DIM_LABELS, _PCA["var"])):
        bar = _pca_bar(v)
        pca_rows.append(f"  PC{i + 1}  {lbl:<32}  {bar}  {v:.1%}")
    pca_html = "".join(
        f'<div class="term-line" style="color:{C["dim"]}">{html.escape(row)}</div>'
        for row in pca_rows
    )
    st.markdown(f'<div class="term-pca-block">{pca_html}</div>', unsafe_allow_html=True)

    st.markdown(
            f'<div class="term-line" style="color:{C["fg"]}">{html.escape("  Answer 10 questions to find your ideal SF group house.")}</div>'
            f'<div class="term-line" style="color:{C["fg"]}">{html.escape("  Pick your favourite word each round.")}</div>',
            unsafe_allow_html=True,
        )

    if n_new > 0:
        st.markdown(
            f'<div class="term-line" style="color:{C["yellow"]}">{html.escape("  Expanding word bank via Datamuse…")}</div>'
            f'<div class="term-line" style="color:{C["green"]}">{html.escape(f"  +{n_new} words fetched  ({total} total in bank).")}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="term-line" style="color:{C["yellow"]}">{html.escape("  Expanding word bank via Datamuse…")}</div>'
            f'<div class="term-line" style="color:{C["pink"]}">{html.escape(f"  (Offline — {total} local words in bank.)")}</div>',
            unsafe_allow_html=True,
        )

    

    st.markdown("")
    if st.button("  ▶  start", key="start_quiz"):
        _reset_quiz()
        st.rerun()


def _render_quiz() -> None:
    nouns = st.session_state.nouns
    hint = st.session_state.hint
    q = st.session_state.q

    st.markdown(
        f'<pre class="term-pre" style="font-weight:700">{html.escape(f"  ── Question {q} / 10 ─────────────────────────")}</pre>',
        unsafe_allow_html=True,
    )
    pick_pre = '  "Pick the word that feels most '
    pick_q = '"'
    st.markdown(
        f'<pre class="term-pre"><span style="font-weight:700">{html.escape(pick_pre)}</span>'
        f'<span style="font-weight:700;color:{C["cyan"]}">you</span>'
        f'<span style="font-weight:700">{html.escape(pick_q)}</span></pre>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    for i, (word, _) in enumerate(nouns):
        label = f"    {i + 1} - {word.capitalize()}"
        if st.button(label, key=f"pick_{q}_{i}"):
            _apply_pick(i)
            st.rerun()


def _render_results() -> None:
    ranked = st.session_state.ranked
    top = ranked[0]

    st.markdown(
        '<div class="term-header"><div class="term-header-inner">'
        f'<div class="term-line" style="color:{C["cyan"]};font-weight:700">'
        f'<div class="term-line" style="color:{C["cyan"]};font-weight:700">'
        f'{html.escape("Results — Houses by Match %")}</div>'
        f'<div class="term-line" style="color:{C["cyan"]};font-weight:700">'
        "</div></div>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    for i, h in enumerate(ranked):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else "  "
        hx = ANSI_TO_HEX.get(h["color"], C["fg"])
        bar_h = _bw_bar_html(h["pct"])
        pct = f"{h['pct']}%"
        st.markdown(
            f'<pre class="term-pre" style="font-family:ui-monospace,Menlo,monospace">'
            f'{html.escape(f"  {medal}  ")}'
            f'<span style="color:{hx};font-weight:700">{html.escape(h["name"])}</span>'
            f'{html.escape("  ")}'
            f"{bar_h}"
            f'{html.escape("  ")}'
            f'<span style="font-weight:700">{html.escape(pct)}</span>'
            f"</pre>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<pre class="term-pre" style="color:{C["dim"]}">       {html.escape(h["desc"])}</pre>',
            unsafe_allow_html=True,
        )
        st.markdown("")

    th = ANSI_TO_HEX.get(top["color"], C["fg"])
    belong = f"  You belong in {top['name']}!"
    st.markdown(
        f'<pre class="term-pre"><span style="font-weight:700;color:{th}">'
        f"{html.escape(belong)}</span></pre>",
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<pre class="term-pre" style="color:{C["dim"]}">{html.escape("  What would you like to do?")}</pre>',
        unsafe_allow_html=True,
    )
    if st.button("  1.  retake quiz", key="retake"):
        _reset_quiz()
        st.rerun()
    if st.button("  2.  back to intro", key="back_intro"):
        st.session_state.phase = "welcome"
        st.rerun()


if __name__ == "__main__":
    main()
