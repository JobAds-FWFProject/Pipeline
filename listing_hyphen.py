"""
Listing and hyphen correction for job-ad text (fast version).

Based on the original notebook
"Final_Notebook_Listing_Hyphen_Problem_29_Newspapers.ipynb", with these
changes to the results:
* "und"/"oder" inside words like "verbunden" are no longer split (_CONN_GLUED_RE),
* line-break hyphens are removed before the listing step
  ("wöchent⸗ lich" -> "wöchentlich", see join_linebreak_hyphens),
* "u." / "od." count as connectors in the hyphen cleanup.
It is also much faster:

* all regular expressions are compiled once at import time,
* suffix lookup normalises each word once and caches the result
  (instead of re-normalising ~250 suffixes for every word),
* the token-by-token hyphen cleanup loop is replaced by one regex
  substitution with identical behaviour,
* rows without any hyphen skip the hyphen-dependent steps entirely,
* rows are processed in parallel with multiple processes (optional),
* no Excel round trip: the data stays in memory.

Usage (see the notebook):

    import pandas as pd
    import listing_hyphen as lh
    df = pd.read_csv("wrz_extractions.csv", index_col=0)
    df = lh.process_dataframe(df, n_jobs=-1)      # shows a progress bar
    df.to_csv("WRZ/wrz_no_lb_listing_hyphen_corrected.csv")
"""
from __future__ import annotations

import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import pandas as pd

# =========================================================
# 1. LINE BREAKS / LIGHT NORMALISATION
# =========================================================

_LINEBREAK_RE = re.compile(r"\s*(?:\r\n|\r|\n)+\s*")
_HSPACE_RE = re.compile(r"[ \t\f\v]+")
# Connectors glued to the neighbouring words, e.g.
#   "Schlosser-undSchmiedegehilfen", "SchlosserundSchmied", "Köchinu.Magd"
# The original notebook matched "und"/"oder" anywhere inside a word, which
# split real words ("verbunden" -> "verb und en", "Grund⸗" -> "Gr und ⸗").
# Now a connector only counts as glued when it is lower case, the next word
# starts with a capital letter (German noun), and it comes directly after a
# hyphen, a space or a lower-case letter ("Taillen⸗ oderPaletotschneider").
_CONN_GLUED_RE = re.compile(r"(?<=[\s⸗\-a-zäöüß])(und|oder|u\.|od\.)(?=[A-ZÄÖÜ])")
_SPACES_RE = re.compile(r"\s+")


def remove_linebreaks(text):
    """Replace line breaks (and surrounding spaces) by one space."""
    if pd.isna(text):
        return text
    text = _LINEBREAK_RE.sub(" ", str(text))
    return _HSPACE_RE.sub(" ", text).strip()


def normalize_text(text):
    """Long s -> s, add spaces around glued connectors, collapse spaces."""
    if pd.isna(text):
        return text
    text = str(text).replace("ſ", "s")
    text = _CONN_GLUED_RE.sub(r" \1 ", text)
    return _SPACES_RE.sub(" ", text).strip()


# Line-break hyphenation: "wöchent⸗ lich" -> "wöchentlich".
# After remove_linebreaks a hyphen at the end of a line is "word⸗ next".
# It is removed when the next word starts with a lower-case letter and is not
# a connector ("Stuben⸗ und Kindermädchen", "Vor⸗ bis Nachmittag" stay).
# Before a capital letter the hyphen is kept ("Kinder- Krankenschwester");
# the hyphen cleanup at the end then removes the space.
_LB_HYPHEN_RE = re.compile(r"(?<=\w)[-⸗]\s+(\w+)")
_KEEP_AFTER_HYPHEN = {
    "und", "oder", "u", "od", "sowie", "bzw", "resp", "beziehungsweise",
    "respektive", "als", "wie", "bis", "noch", "zu",
}


def _join_lb_hyphen(m):
    word = m.group(1)
    if word[0].islower() and word not in _KEEP_AFTER_HYPHEN:
        return word
    return m.group(0)


def join_linebreak_hyphens(text):
    if pd.isna(text):
        return text
    if "-" not in text and "⸗" not in text:
        return text
    return _LB_HYPHEN_RE.sub(_join_lb_hyphen, text)


def normalize_for_matching(text):
    if text is None:
        return text
    return (
        str(text).lower()
        .replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
    )

# =========================================================
# 2. CURATED SUFFIX LIST
# =========================================================

OCCUPATION_SUFFIXES = [
    # generic positions / roles
    "hilfskräfte", "hilfskraft",
    "aushilfen", "aushilfe",
    "angestellten", "angestellte", "angestellter",
    "sekretärinnen", "sekretärin", "sekretäre", "sekretär",
    "buchhalterinnen", "buchhalterin", "buchhalter",
    "kassiererinnen", "kassiererin", "kassierer", "kassierin", "kassier",
    "lehrerinnen", "lehrerin", "lehrer",
    "assistentinnen", "assistentin", "assistenten", "assistent",
    "konstrukteurinnen", "konstrukteurin", "konstrukteure", "konstrukteur",
    "beamtinnen", "beamtin", "beamten", "beamter",
    "praktikantinnen", "praktikantin", "praktikanten", "praktikant",
    "kontoristinnen", "kontoristin", "kontoristen", "kontorist",
    "inkassantinnen", "inkassantin", "inkassanten", "inkassant",
    "magazineurinnen", "magazineurin", "magazineure", "magazineur",
    "schreiberinnen", "schreiberin", "schreiber",
    "stellen", "stelle", "posten", "ämter", "amt",
    "dienste", "dienst",
    "personal", "kräfte", "kraft",

    # worker / trades
    "arbeiterinnen", "arbeiterin", "arbeiter",
    "meisterinnen", "meisterin", "meister",
    "gehilfinnen", "gehilfin", "gehilfen", "gehilfe",
    "gehülfinnen", "gehülfin", "gehülfen", "gehülfe",

    "schlosserinnen", "schlosserin", "schlosser",
    "schmiedinnen", "schmiedin", "schmiede", "schmied",
    "tischlerinnen", "tischlerin", "tischler",
    "zimmerleute", "zimmermann",
    "bäckerinnen", "bäckerin", "bäcker",
    "fleischerinnen", "fleischerin", "fleischer",
    "müllerinnen", "müllerin", "müller",
    "weberinnen", "weberin", "weber",
    "druckerinnen", "druckerin", "drucker",
    "schneiderinnen", "schneiderin", "schneider",
    "sattlerinnen", "sattlerin", "sattler",
    "glaserinnen", "glaserin", "glaser",
    "malerinnen", "malerin", "maler",
    "lackiererinnen", "lackiererin", "lackierer",
    "dreherinnen", "dreherin", "dreher",
    "vergolderinnen", "vergolderin", "vergolder",
    "schweißerinnen", "schweißerin", "schweißer",
    "schweisserinnen", "schweisserin", "schweisser",

    # irregular / food / textile
    "köchinnen", "köchin", "koch", "köche",
    "näherinnen", "näherin", "näher", "naherin", "naherinnen",
    "wäscherinnen", "wäscherin", "wäscher",
    "plätterinnen", "plätterin", "plätter",
    "büglerinnen", "büglerin", "bügler",
    "stickerinnen", "stickerin", "sticker",

    # domestic
    "haushälterinnen", "haushälterin", "haushälter",
    "mägde", "magd",
    "frauen", "frau", "fräulein",
    "mädchen",

    # service
    "dienerinnen", "dienerin", "diener",
    "wärterinnen", "wärterin", "wärter",
    "pförtnerinnen", "pförtnerin", "pförtner",
    "heizerinnen", "heizerin", "heizer",
    "kutscherinnen", "kutscherin", "kutscher",
    "fahrerinnen", "fahrerin", "fahrer",
    "botinnen", "botin", "boten", "bote",
    "trägerinnen", "trägerin", "träger",
    "kellnerinnen", "kellnerin", "kellner",
    "burschen", "bursche",
    "junge", "jungen",

    # legal / clerical / administrative
    "kanzlistinnen", "kanzlistin", "kanzlisten", "kanzlist",

    # medical / agrarian
    "ärztinnen", "ärztin", "ärzte", "arzt",
    "meierinnen", "meierin", "meier",

    # training / generic
    "lehrlinge", "lehrling",
    "eleven", "eleve",

    # ---- added after reviewing missed listings in wrz + nwj ----
    # management / office / sales
    "führerinnen", "führerin", "führer",
    "leiterinnen", "leiterin", "leiter",
    "verwalterin", "verwalter",
    "aufseherin", "aufseher",
    "inspector", "inspektor", "director", "direktor",
    "portierin", "portiere", "portier",
    "cassierin", "cassier", "kassierer",
    "bibliothekar", "administrator", "chauffeur",
    "correspondenten", "korrespondenten",
    "besitzerinnen", "besitzerin", "besitzer",
    "verkäuferinnen", "verkäuferin", "verkäufer",
    "händlerinnen", "händlerin", "händler",
    "vertreterin", "vertreter",
    "reisender", "reisende",
    "agenten", "agent",
    "commis", "kommis",
    "correspondentin", "correspondent", "korrespondentin", "korrespondent",
    "expedientin", "expedient",
    "kaufleute", "kaufmann",
    "sammlerin", "sammler",
    "beamte",
    # technical / trades
    "zeichnerinnen", "zeichnerin", "zeichner",
    "monteure", "monteur",
    "techniker", "mechaniker", "chemiker",
    "ingenieure", "ingenieur",
    "macherinnen", "macherin", "macher",
    "binderin", "binder",
    "gießer", "hauer",
    "gesellen", "geselle",
    # service / domestic / care
    "besorgerinnen", "besorgerin", "besorger",
    "knechte", "knecht",
    "bursch",
    "putzerinnen", "putzerin", "putzer",
    "wächterin", "wächter",
    "pflegerinnen", "pflegerin", "pfleger",
    "schwestern", "schwester",
    "gärtnerinnen", "gärtnerin", "gärtner",
    "erzieherinnen", "erzieherin", "erzieher",
    "wirtschafterinnen", "wirtschafterin", "wirthschafterinnen", "wirthschafterin",
    "modistinnen", "modistin",
    "friseurinnen", "friseurin", "friseure", "friseur", "friseusen", "friseuse",
    # music / school
    "spielerinnen", "spielerin", "spieler",
    "schülerinnen", "schülerin", "schülern", "schüler",
]

# Longest suffix first (ties broken alphabetically so the order is reproducible).
OCCUPATION_SUFFIXES = sorted(set(OCCUPATION_SUFFIXES), key=lambda s: (-len(s), s))
_SUFFIXES_NORM = [(normalize_for_matching(s), s) for s in OCCUPATION_SUFFIXES]
_SUFFIX_SET = set(OCCUPATION_SUFFIXES)

# =========================================================
# 3. LISTING REGEXES
# =========================================================

HYPHEN = r"(?:-|⸗)"
CONNECTOR_RE = r"(?:und|oder|u\.|od\.|&)"

# Note: the original had "\s*[,;:]?\s*" here. Without a comma the two \s* can
# share a space in two ways, so a run of n "word- " pieces without a connector
# was tried 2**n times (hung for hours on OCR junk like "einrich⸗ tungseinrich⸗ ...").
# "\s*(?:[,;:]\s*)?" matches exactly the same text without that ambiguity.
LISTING_PATTERN = re.compile(
    rf"""
    (?P<listing>
        (?P<pre>
            (?:\b\w+{HYPHEN}\s*(?:[,;:]\s*)?)+
        )
        (?P<conn>{CONNECTOR_RE})
        \s*
        (?P<after>
            (?:\b\w+{HYPHEN}\s*)*
            \w+(?:\s+\w+){{0,1}}
        )
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PREFIX_RE = re.compile(rf"\b(\w+){HYPHEN}")
_BROKEN_PIECE_RE = re.compile(rf"\s*((?:\b\w+{HYPHEN}\s*)+)(\w+)\s*", re.IGNORECASE)
_SPLIT_WORD_RE = re.compile(rf"\s*(\w+){HYPHEN}\s+(\w+)\s*", re.IGNORECASE)
_WORD_RE = re.compile(r"\w+")
_NONSPACE_RE = re.compile(r"\S+")

STOP_TOKENS = {
    "suche", "sucht", "gesucht", "suchte", "suchen",
    "aufgenommen", "aufnahme",
    "wird", "werden", "wurde",
    "findet", "finden", "fand",
    "ist", "sind", "war", "waren",
}
CONNECTOR_TOKENS = {"und", "oder", "u.", "od.", "&"}

# =========================================================
# 4. SUFFIX EXTRACTION
# =========================================================


@lru_cache(maxsize=200_000)
def extract_suffix_from_word(word):
    """Return (stem, suffix) for the longest known occupation suffix, else (None, None)."""
    word_norm = normalize_for_matching(word)
    for suffix_norm, suffix in _SUFFIXES_NORM:
        if word_norm.endswith(suffix_norm):
            return word[:-len(suffix)], suffix
    return None, None


def extract_suffix_from_phrase(phrase):
    words = _WORD_RE.findall(phrase)
    if not words:
        return None, None, None, []
    for i in range(len(words) - 1, -1, -1):
        stem, suffix = extract_suffix_from_word(words[i])
        if suffix is not None:
            return i, stem, suffix, words
    return None, None, None, words

# =========================================================
# 5. RIGHT-HAND SIDE PARSING
# =========================================================


def parse_after_as_pure_broken_piece(after):
    m = _BROKEN_PIECE_RE.fullmatch(after)
    if not m:
        return None, None
    prefixes = _PREFIX_RE.findall(m.group(1))
    if not prefixes:
        return None, None
    return prefixes, m.group(2)


def parse_after_as_split_single_word(after):
    m = _SPLIT_WORD_RE.fullmatch(after)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(1) + m.group(2)


def clean_token_for_boundary_check(token):
    return token.strip().lower().strip(",;:.!?")


def get_local_after_candidate(after):
    token_matches = list(_NONSPACE_RE.finditer(after))
    if not token_matches:
        return "", ""

    local = []
    for m in token_matches:
        tok_clean = clean_token_for_boundary_check(m.group(0))
        if tok_clean in STOP_TOKENS or tok_clean in CONNECTOR_TOKENS:
            break
        local.append(m)

    if not local:
        return "", after.strip()

    local_span_end = local[-1].end()
    local_span = after[:local_span_end].strip()

    broken_prefixes, _ = parse_after_as_pure_broken_piece(local_span)
    if broken_prefixes is not None:
        return local_span, after[local_span_end:].lstrip()

    # OCR-split word: "Hausbesorger stelle" -> keep both tokens so that the
    # ending is "stelle" ("Portier⸗ od. Hausbesorger stelle" -> "Portierstelle").
    if len(local) >= 2:
        second = local[1].group(0)
        if (second[0].islower()
                and clean_token_for_boundary_check(second) in _SUFFIX_SET):
            end = local[1].end()
            return after[:end].strip(), after[end:].lstrip()

    first_word_match = _WORD_RE.search(local[0].group(0))
    if first_word_match:
        _, first_suffix = extract_suffix_from_word(first_word_match.group(0))
        if first_suffix is not None:
            end = local[0].end()
            return after[:end].strip(), after[end:].lstrip()

    end = local[min(2, len(local)) - 1].end()
    return after[:end].strip(), after[end:].lstrip()

# =========================================================
# 6. RECONSTRUCTION
# =========================================================


def join_items_with_connector(items, connector_text):
    connector_norm = connector_text.strip()
    if len(items) == 0:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} {connector_norm} {items[1]}"
    return ", ".join(items[:-1]) + f" {connector_norm} {items[-1]}"


def _rebuild(items, connector, remainder_after):
    deduped = list(dict.fromkeys(k for k in (i.strip() for i in items) if k))
    rebuilt = join_items_with_connector(deduped, connector)
    if remainder_after:
        rebuilt = rebuilt + " " + remainder_after
    return rebuilt


def _ending_from_other_noun(candidate, words, suffix_index):
    """
    True if the ending was found in a capitalised word that follows another
    capitalised word, e.g. "Früh⸗ oder Nachmittagsstunden Stelle": the ending
    "stelle" belongs to a different noun, so "Frühstelle" would be wrong.

    Still allowed:
      * adjective before the noun:  "Aufseher⸗ oder kleineren Verwalterposten"
      * OCR-split word (lower case): "Portier⸗ od. Hausbesorger stelle"
      * hyphenated compound:         "Mandoline⸗ und Guitarre⸗Lehrer"
    """
    if suffix_index == 0 or not words[suffix_index][0].isupper():
        return False
    seen = 0
    for token in candidate.split():          # whitespace tokens; "Guitarre⸗Lehrer" is one
        token_words = _WORD_RE.findall(token)
        if seen + len(token_words) > suffix_index:   # token holding the ending
            return False
        if token_words and token_words[0][0].isupper():
            return True
        seen += len(token_words)
    return False


_COMPOUND_RE = re.compile(r"(\w+)([-⸗])(\w+)")
_GENERIC_SUFFIXES = {"stelle", "stellen", "posten", "amt", "ämter", "dienst", "dienste",
                     "personal", "kraft", "kräfte"}
# Only these occupations are copied together with a generic ending; with
# others the guess was often wrong in the wrz/nwj review.
_SHAREABLE_OCCUPATIONS = {"diener", "dienerin", "leiter", "leiterin", "arbeiter", "arbeiterin"}


def _shared_occupation(word, suffix):
    """
    "Geschäftsdienerposten" with ending "posten" -> "diener", so that
    "Bureau⸗ oder Geschäftsdienerposten" -> "Bureaudienerposten" (not "Bureauposten").
    The caller skips left words that are occupations themselves
    ("Hilfsarbeiter⸗ oder Geschäftsdienerposten" -> "Hilfsarbeiterposten").
    """
    if suffix not in _GENERIC_SUFFIXES:
        return None
    stem = word[:-len(suffix)]
    if not stem:
        return None
    _, occ = extract_suffix_from_word(stem)
    return occ if occ in _SHAREABLE_OCCUPATIONS else None


def reconstruct_listing(match):
    original = match.group("listing")
    connector = match.group("conn")
    after = match.group("after").strip()

    pre_prefixes = _PREFIX_RE.findall(match.group("pre"))
    if not pre_prefixes:
        return original

    candidate_after, remainder_after = get_local_after_candidate(after)
    if not candidate_after:
        return original

    # a) one word split by a hyphen, e.g. "Schnei- derinnen"
    split_prefix, rest_of_word, repaired = parse_after_as_split_single_word(candidate_after)
    if repaired is not None and rest_of_word[0].isupper():
        # Not a broken word but a compound: "Eisenbahn⸗ Oberbeamter",
        # "Restaurant⸗ Angestellter" -> keep the hyphen, same rule as below.
        hyphen = candidate_after[len(split_prefix)]
        repaired = split_prefix + hyphen + rest_of_word
        _, suffix = extract_suffix_from_word(rest_of_word)
        if suffix is None:
            return original
        if normalize_for_matching(rest_of_word) != normalize_for_matching(suffix):
            items = [p + hyphen + rest_of_word for p in pre_prefixes] + [repaired]
        else:
            items = [p + suffix for p in pre_prefixes] + [repaired]
        return _rebuild(items, connector, remainder_after)
    if repaired is not None:
        _, suffix = extract_suffix_from_word(repaired)
        if suffix is None:
            return original
        items = [p + suffix for p in pre_prefixes] + [repaired]
        return _rebuild(items, connector, remainder_after)

    # b) pure broken right-hand piece, e.g. "Maschinen- Näherinnen"
    after_prefixes, after_final_word = parse_after_as_pure_broken_piece(candidate_after)
    if after_prefixes is not None:
        _, suffix = extract_suffix_from_word(after_final_word)
        if suffix is None:
            return original
        # Hyphenated compound whose second part is more than the ending:
        # "Bau- und Galanterie-Spenglermeister"
        #   -> "Bau-Spenglermeister und Galanterie-Spenglermeister"
        # (not "Baumeister"). A bare ending ("Ajour⸗Näherinnen") keeps the
        # normal behaviour ("Handnäherinnen").
        m = _COMPOUND_RE.fullmatch(candidate_after)
        if (m and after_final_word[0].isupper()
                and normalize_for_matching(after_final_word) != normalize_for_matching(suffix)):
            items = [p + m.group(2) + after_final_word for p in pre_prefixes] + [candidate_after]
            return _rebuild(items, connector, remainder_after)
        items = [p + suffix for p in pre_prefixes] + [p + suffix for p in after_prefixes]
        return _rebuild(items, connector, remainder_after)

    # c) normal local candidate (1-2 words)
    suffix_index, _, suffix, words = extract_suffix_from_phrase(candidate_after)
    if suffix is None:
        return original
    if _ending_from_other_noun(candidate_after, words, suffix_index):
        return original
    shared = _shared_occupation(words[suffix_index], suffix)
    items = [p + (shared if shared and extract_suffix_from_word(p)[1] is None else "") + suffix
             for p in pre_prefixes] + [candidate_after]
    return _rebuild(items, connector, remainder_after)


def correct_listings(text):
    if pd.isna(text):
        return text
    if "-" not in text and "⸗" not in text:   # pattern needs a hyphen
        return text
    return LISTING_PATTERN.sub(reconstruct_listing, text)

# =========================================================
# 7. HYPHEN-SPACE CLEANUP
# =========================================================

# Matches can only start at the beginning of a word, so (?<!\w) does not change
# the result but avoids retrying from every character inside a word.
# Possessive quantifiers (++, *+; Python >= 3.11) stop useless backtracking;
# they are equivalent here because a word must be followed by space/hyphen.
try:
    _CHAIN_RE = re.compile(r"(?<!\w)(\w++)\s*+(-|⸗)\s+(\w++)\s*+(-|⸗)\s*(\w+)")
    _HYPHEN_PAIR_RE = re.compile(r"(?<!\w)(\w++)\s*+(-|⸗)\s*(\w+)")
except re.error:  # Python < 3.11
    _CHAIN_RE = re.compile(r"(?<!\w)(\w+)\s*(-|⸗)\s+(\w+)\s*(-|⸗)\s*(\w+)")
    _HYPHEN_PAIR_RE = re.compile(r"(?<!\w)(\w+)\s*(-|⸗)\s*(\w+)")
_CONNECTOR_WORDS = {"und", "oder"}


def _is_word_char(c):
    return c.isalnum() or c == "_"   # same definition as regex \w


def _prev_token_is_connector(text, pos):
    """Is the last non-space token before `pos` 'und', 'oder' or '&'?"""
    end = pos
    while end > 0 and text[end - 1].isspace():
        end -= 1
    if end == 0:
        return False
    if text[end - 1] == "&":
        return True
    for word in ("und", "oder"):
        start = end - len(word)
        if (start >= 0 and text[start:end].lower() == word
                and (start == 0 or not _is_word_char(text[start - 1]))):
            return True
    return False


def clean_hyphen_spaces_carefully(text):
    """
    Remove spaces around hyphens inside compounds, e.g.
        "Kettenstich- Maschin ⸗Näherinnen" -> "Kettenstich-Maschin⸗Näherinnen"
    but keep listings intact ("Schlosser- und Schmiedegehilfen", "und Schlosser- ...").
    """
    if pd.isna(text):
        return text
    text = str(text)
    if "-" not in text and "⸗" not in text:
        return text

    # Step A: hyphen chains, repeated until nothing changes
    prev = None
    while prev != text:
        prev = text
        text = _CHAIN_RE.sub(r"\1\2\3\4\5", text)

    # Step B: join "word - word" unless a connector is on either side
    def _join(m):
        left, hyphen, right = m.group(1), m.group(2), m.group(3)
        r = right.lower()
        if (r in _KEEP_AFTER_HYPHEN   # und, oder, u., od., bis, sowie, ...
                or _prev_token_is_connector(text, m.start())):
            return m.group(0)
        return left + hyphen + right

    return _HYPHEN_PAIR_RE.sub(_join, text)

# =========================================================
# 8. ROW / DATAFRAME PROCESSING (PARALLEL)
# =========================================================


def _process_texts(texts):
    """All steps for a list of raw `post_corrected` strings (one worker's chunk)."""
    out = []
    for raw in texts:
        no_lb = remove_linebreaks(raw)
        norm = join_linebreak_hyphens(normalize_text(no_lb))
        listings = correct_listings(norm)
        final = clean_hyphen_spaces_carefully(listings)
        out.append((no_lb, norm, listings, final))
    return out


def _n_workers(n_jobs):
    if n_jobs is None or n_jobs == 1:
        return 1
    if n_jobs > 0:
        return n_jobs
    try:  # respects SLURM / cgroup CPU allocation on Linux
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class ProgressBar:
    """
    Plain-text progress bar, e.g.
        [##########----------]  50%  246,000/492,481 rows  0:08 elapsed, ~0:08 left
    It only prints text and redraws the line with "\\r", so it works in
    Jupyter on the cluster without widgets (where tqdm shows an empty bar).
    """

    def __init__(self, total, width=30, min_interval=0.5, file=None):
        self.total, self.width, self.min_interval = total, width, min_interval
        self.file = file or sys.stdout
        self.done = 0
        self.start = self._last = time.perf_counter()
        # Speed is measured from the first finished chunk on, so the time to
        # start the worker processes does not distort the estimate.
        self._t_first = None
        self._done_first = 0

    @staticmethod
    def _fmt(seconds):
        seconds = int(round(seconds))
        h, rest = divmod(seconds, 3600)
        m, s = divmod(rest, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"

    def _draw(self):
        frac = self.done / self.total if self.total else 1.0
        filled = int(self.width * frac)
        now = time.perf_counter()
        elapsed = now - self.start
        line = (f"[{'#' * filled}{'-' * (self.width - filled)}] {frac:4.0%}  "
                f"{self.done:,}/{self.total:,} rows  {self._fmt(elapsed)} elapsed")
        if frac < 1 and self._t_first is not None and self.done > self._done_first:
            rate = (self.done - self._done_first) / (now - self._t_first)
            line += f", ~{self._fmt((self.total - self.done) / rate)} left"
        self.file.write("\r" + line + "  ")
        self.file.flush()

    def update(self, n):
        self.done += n
        now = time.perf_counter()
        if self._t_first is None:
            self._t_first, self._done_first = now, self.done
        if now - self._last >= self.min_interval or self.done >= self.total:
            self._last = now
            self._draw()

    def close(self):
        if self.done < self.total or self.total == 0:
            self._draw()
        self.file.write("\n")
        self.file.flush()


def process_dataframe(df, text_column="post_corrected", n_jobs=-1, chunksize=2000,
                      progress=True):
    """
    Return a copy of `df` with the columns of the original notebook added:
        post_corrected_without_lb, text_norm, post_corrected_listings,
        listing_changed, post_corrected_final, hyphen_space_changed
    (text_norm also has the line-break hyphens removed.)

    n_jobs:   number of processes (-1 = all available CPUs, 1 = no multiprocessing).
    progress: show a plain-text progress bar.
    Reading and writing files is left to the caller (see the notebook).
    """
    df = df.copy()
    texts = df[text_column].tolist()
    workers = min(_n_workers(n_jobs), max(1, len(texts) // chunksize))
    chunks = [texts[i:i + chunksize] for i in range(0, len(texts), chunksize)]
    bar = ProgressBar(len(texts)) if progress else None

    results = []
    if workers == 1:
        chunk_results = map(_process_texts, chunks)
        for chunk in chunk_results:
            results.extend(chunk)
            if bar:
                bar.update(len(chunk))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for chunk in ex.map(_process_texts, chunks):   # keeps row order
                results.extend(chunk)
                if bar:
                    bar.update(len(chunk))
    if bar:
        bar.close()

    no_lb, norm, listings, final = (list(c) for c in zip(*results)) if results else ([],) * 4
    df["post_corrected_without_lb"] = no_lb
    df["text_norm"] = norm
    df["post_corrected_listings"] = listings
    df["listing_changed"] = ~df["text_norm"].fillna("").eq(
        df["post_corrected_listings"].fillna(""))
    df["post_corrected_final"] = final
    df["hyphen_space_changed"] = ~df["post_corrected_listings"].fillna("").eq(
        df["post_corrected_final"].fillna(""))
    return df
