"""
chunk_corpus.py - offline, structure-aware chunker for the regulatory corpus.

Reads corpus/raw/*.html (or *.txt), strips HTML, drops site-nav boilerplate,
splits on CFR/USC paragraph markers (and falls back to sentence splitting for
prose sources like CFPB circulars), and writes corpus/chunks.jsonl.

Each chunk carries a `citation` parsed from the filename + nearest marker, which
is what makes retrieval legally groundable and what the RAG-vs-noRAG
"legal grounding" eval scores against.

No network. Only dep: beautifulsoup4 (pip install beautifulsoup4).
Run:  python corpus/chunk_corpus.py
"""

import json, re, sys
from pathlib import Path

try:
    from bs4 import BeautifulSoup
except ImportError:
    sys.exit("pip install beautifulsoup4")

RAW = Path(__file__).parent / "raw"
OUT = Path(__file__).parent / "chunks.jsonl"

CITATIONS = {
    "reg_b_1002_2":          "12 CFR 1002.2",
    "reg_b_1002_9":          "12 CFR 1002.9",
    "reg_b_appendix_a":      "12 CFR pt.1002 App. A",
    "reg_b_appendix_c":      "12 CFR pt.1002 App. C",
    "reg_b_supplement_i_9":  "12 CFR pt.1002 Supp. I (comment 9)",
    "fcra_615":              "15 USC 1681m (FCRA 615)",
    "cfpb_circular_2022_03": "CFPB Circular 2022-03",
    "cfpb_circular_2023_03": "CFPB Circular 2023-03",
}

TARGET_CHARS = 1600      # ~350-450 tokens; keeps one requirement per chunk
MAX_CHARS    = 2400      # ceiling before force-splitting a long block
MIN_CHARS    = 150       # drop anything shorter than this (nav scraps)

# Phrases that mark site chrome / boilerplate. A chunk that is mostly these
# (or that starts with one) is dropped.
JUNK_MARKERS = [
    "skip to main content", "please help us improve", "quick search by citation",
    "no thank you", "sign in / sign up", "table of popular names",
    "site feedback", "reader aids", "you are using an unsupported browser",
    "cornell law school", "search cornell", "toggle navigation",
    "enhanced content", "this content is from the ecfr",
]

MARKER = re.compile(r"^\s*\(([a-zA-Z0-9]{1,4})\)\s")   # (a) (1) (i) (A)
SECTION = re.compile(r"§?\s*(\d{3,4}\.\d+[a-z0-9()]*)")
SENTENCE = re.compile(r"(?<=[.;:])\s+(?=[A-Z0-9(\u201c\"])")  # split prose on sentence ends


def load_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in (".html", ".htm"):
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "form", "button", "aside"]):
            tag.decompose()
        raw = soup.get_text("\n")
    lines = [ln.strip() for ln in raw.splitlines()]
    return "\n".join(ln for ln in lines if ln)


def is_junk(text: str) -> bool:
    if len(text.strip()) < MIN_CHARS:
        return True
    low = text.lower()
    hits = sum(1 for m in JUNK_MARKERS if m in low)
    # drop if it opens with boilerplate or is riddled with it
    if any(low.lstrip().startswith(m) for m in JUNK_MARKERS):
        return True
    return hits >= 2


def blocks(text: str):
    for para in text.split("\n"):
        if len(para) < 3:
            continue
        m = MARKER.match(para)
        yield (m.group(1) if m else None), para


def split_long_prose(text: str):
    """Split a marker-less block into ~TARGET_CHARS pieces on sentence bounds."""
    sents, buf, out = SENTENCE.split(text), [], []
    cur = 0
    for s in sents:
        buf.append(s)
        cur += len(s) + 1
        if cur >= TARGET_CHARS:
            out.append(" ".join(buf)); buf, cur = [], 0
    if buf:
        out.append(" ".join(buf))
    return out


def chunk_file(path: Path):
    stem = path.stem
    base_cite = CITATIONS.get(stem, stem)
    buf, buf_len, buf_marker = [], 0, None
    raw_chunks = []

    def flush():
        nonlocal buf, buf_len, buf_marker
        if not buf:
            return
        text = " ".join(buf).strip()
        # if this accumulated block is huge and marker-less, sentence-split it
        pieces = [text]
        if len(text) > MAX_CHARS and buf_marker is None:
            pieces = split_long_prose(text)
        for piece in pieces:
            sec = SECTION.search(piece)
            cite = base_cite
            if sec and sec.group(1) not in base_cite:
                cite = f"{base_cite} [{sec.group(1)}]"
            elif buf_marker:
                cite = f"{base_cite}({buf_marker})"
            raw_chunks.append({"source": stem, "citation": cite, "text": piece})
        buf, buf_len, buf_marker = [], 0, None

    for marker, para in blocks(load_text(path)):
        if buf and (buf_len + len(para) > TARGET_CHARS) and marker:
            flush()
        if buf_marker is None and marker:
            buf_marker = marker
        buf.append(para)
        buf_len += len(para) + 1
        if buf_len >= MAX_CHARS:
            flush()
    flush()

    # drop nav/boilerplate chunks
    return [c for c in raw_chunks if not is_junk(c["text"])]


def main():
    if not RAW.exists():
        sys.exit(f"missing {RAW} - save the sources first (see corpus/README.md)")
    files = sorted(p for p in RAW.iterdir()
                   if p.suffix.lower() in (".html", ".htm", ".txt"))
    if not files:
        sys.exit(f"no files in {RAW}")

    all_chunks = []
    for p in files:
        c = chunk_file(p)
        all_chunks.extend(c)
        print(f"{p.name:32s} -> {len(c):3d} chunks")

    with OUT.open("w", encoding="utf-8") as f:
        for i, ch in enumerate(all_chunks):
            ch["chunk_id"] = f"c{i:04d}"
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")

    avg = sum(len(c["text"]) for c in all_chunks) / max(len(all_chunks), 1)
    print(f"\nwrote {len(all_chunks)} chunks -> {OUT}")
    print(f"avg chunk length: {avg:.0f} chars (~{avg/4:.0f} tokens)")


if __name__ == "__main__":
    main()
