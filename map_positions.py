"""Map `position` spans (offsets into `post_corrected_final`) back onto `post_corrected`.

`post_corrected_final` is a normalised rewrite of `post_corrected`:
  - long s (ſ) -> s, line breaks -> spaces, end-of-line hyphenation removed
  - elliptical listings expanded, e.g. "Geſchäfts⸗ oder Hausdiener" -> "Geschäftsdiener oder Hausdiener"

Both texts are aligned character-wise and every span is projected onto
`post_corrected`, so "Geschäftsdiener" becomes "Geſchäfts⸗" with its original offsets.

Notebook usage:
    import pandas as pd
    from map_positions import add_pc_positions

    df = pd.read_csv("nwj_extractions.csv", index_col=0)
    df["position_post_corrected"] = add_pc_positions(df)
"""
import ast
import os
import re
import time
from bisect import bisect_right
from difflib import SequenceMatcher
from multiprocessing import Pool

import pandas as pd

HYPHENS = "⸗-¬="
_CONJ = r"(und|oder|u\.|o\.|sowie|bis|resp\.)"
CONJ = re.compile(_CONJ + r"\b")
# "Berg⸗ und" when the final text turned "Berg⸗ und Hütten…" into "Bergstelle, Hüttenstelle"
TRAILING_CONJ = re.compile(r"[" + re.escape(HYPHENS) + r"]\s+" + _CONJ + r"$")
ENDS_WITH_CONJ = re.compile(r"\b" + _CONJ + r"$")


def normalise(orig):
    """Approximate the post_corrected -> post_corrected_final rewrite.

    Returns (norm, idx): idx[k] is the index in `orig` of norm[k], plus a sentinel len(orig).
    """
    out, idx = [], []
    i, n = 0, len(orig)
    while i < n:
        ch = orig[i]
        if ch in HYPHENS:  # line-break hyphenation: hyphen + whitespace containing a newline
            j = i + 1
            while j < n and orig[j].isspace():
                j += 1
            if "\n" in orig[i + 1 : j] and j < n and not CONJ.match(orig, j):
                if not orig[j].islower():  # Kunstblumen⸗\nArbeiterinnen keeps its hyphen
                    out.append(ch)
                    idx.append(i)
                i = j  # Kund⸗\nstrickerinnen -> Kundstrickerinnen
                continue
        if ch.isspace():  # whitespace run -> single space
            out.append(" ")
            idx.append(i)
            while i < n and orig[i].isspace():
                i += 1
            continue
        out.append("s" if ch == "ſ" else ch)
        idx.append(i)
        i += 1
    idx.append(n)
    return "".join(out), idx


def map_spans(spans, final, orig):
    """Map [(start, end, text), ...] from `final` onto `orig`."""
    norm, idx = normalise(orig)
    ops = [op for op in SequenceMatcher(None, final, norm, autojunk=False).get_opcodes() if op[1] != op[2]]
    starts = [op[1] for op in ops]

    def to_norm(pos, is_end):
        p = pos - 1 if is_end else pos  # end is exclusive: map the last character
        tag, i1, _, j1, j2 = ops[bisect_right(starts, p) - 1]
        if tag != "equal":
            return j2 if is_end else j1
        return j1 + (p - i1) + is_end

    out = []
    for s, e, _ in spans:
        if e <= s:
            out.append((s, e, ""))
            continue
        a, b = to_norm(s, False), to_norm(e, True)
        ns = idx[a]
        ne = idx[b - 1] + 1 if b > a else ns
        while ns < ne and orig[ns].isspace():
            ns += 1
        while ne > ns and orig[ne - 1].isspace():
            ne -= 1
        # keep an elliptical hyphen directly after the span, unless it also follows it in `final`
        if ns < ne < len(orig) and orig[ne] in HYPHENS and (e >= len(final) or final[e] not in HYPHENS):
            ne += 1
        # drop a conjunction pulled in after the hyphen, unless the final span has one too
        m = TRAILING_CONJ.search(orig, ns, ne)
        if m and not ENDS_WITH_CONJ.search(final[s:e]):
            ne = m.start() + 1
        out.append((ns, ne, orig[ns:ne]))
    return out


def _worker(args):
    spans, final, orig = args
    as_str = isinstance(spans, str)
    parsed = ast.literal_eval(spans) if as_str else spans
    mapped = map_spans(parsed, final, orig) if parsed else []
    return repr(mapped) if as_str else mapped


def _has_spans(v):
    return isinstance(v, (str, list)) and v not in ("", "[]") and len(v) > 0


def add_pc_positions(df, position_col="position", final_col="post_corrected_final",
                     orig_col="post_corrected", n_jobs=None, report_every=30):
    """Return a Series with `position_col` mapped onto `orig_col`, aligned to df.index.

    Spans may be strings like "[(0, 15, 'x')]" (as read from CSV) or lists of tuples;
    the output uses the same form. n_jobs=1 runs without multiprocessing.
    Progress is printed every `report_every` seconds.
    """
    result = df[position_col].astype(object)
    todo = df.index[df[position_col].map(_has_spans)]
    jobs = list(zip(df.loc[todo, position_col], df.loc[todo, final_col].fillna(""), df.loc[todo, orig_col].fillna("")))
    total, t0, last, out = len(jobs), time.time(), time.time(), []
    n_jobs = n_jobs or os.cpu_count()
    print(f"Mapping {total:,} rows with spans using {n_jobs} process(es)", flush=True)

    def progress(force=False):
        nonlocal last
        now = time.time()
        if force or now - last >= report_every:
            last, done, el = now, len(out), now - t0
            eta = el / done * (total - done) if done else 0
            print(f"  {done:,}/{total:,} ({done / max(total, 1):.0%}) | {el / 60:.1f} min elapsed | ETA {eta / 60:.1f} min", flush=True)

    if n_jobs == 1:
        for job in jobs:
            out.append(_worker(job))
            progress()
    else:
        with Pool(n_jobs) as pool:
            for r in pool.imap(_worker, jobs, chunksize=200):
                out.append(r)
                progress()
    progress(force=True)
    result.loc[todo] = pd.Series(out, index=todo, dtype=object)
    return result
