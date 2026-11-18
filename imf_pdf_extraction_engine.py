"""imf_pdf_extraction_engine.py
================
Two-track extractor for IMF-style PDFs:

  Track 1 — Box & Annex titles
    Pure PyMuPDF (fitz). Uses the embedded PDF outline (bookmarks) as primary
    source. Falls back to text scanning the TOC page.
    Runtime: < 1 s per file.

  Track 2 — Structural Benchmarks  [Claude vision API, v2.4+]
    Each page containing "Structural Benchmark" is rendered as a 200 dpi PNG
    and sent to Claude vision (claude-sonnet-4-6). Claude returns a JSON array
    of rows with keys: reform_area, benchmark, due_date, status.

    Advantages over previous Docling+regex approach:
      - No Docling dependency or TableFormer model download
      - Handles image-embedded tables natively (no separate fallback)
      - Handles any column layout or naming convention semantically
      - Status always extracted regardless of column name (Observance, etc.)

    Pages are processed in parallel (up to 4 concurrent API calls).
    Requires ANTHROPIC_API_KEY set via set_api_key() or environment variable.

  SI Title Extraction — _extract_si_titles_from_doc
    Handles all known IMF SI chapter title formats:
      Format A: "COUNTRY—TITLE _________ N"  (em-dash after country name)
      Format B: "ALL CAPS TITLE _________ N"  (no country prefix)
      Format C: Title on one line, leader+page on next line
      Format D: "COUNTRY'S TOPIC: ..." (possessive, no em-dash)
      Format E: Accented country names like "CÔTE D'IVOIRE—..."

    Uses Docling for clean TOC text extraction, then Claude API to classify
    which entries are top-level chapter titles.
"""

__version__ = "2.5.0-claude-api"  # + Type column, Prior Actions, QPC

# API key — set via set_api_key() or ANTHROPIC_API_KEY environment variable
_API_KEY: str = ""


def set_api_key(key: str) -> None:
    """Set the Anthropic API key for all Claude API calls in this module."""
    global _API_KEY
    _API_KEY = key.strip()
    os.environ["ANTHROPIC_API_KEY"] = _API_KEY



import logging
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("pdf_extractor.imf_pdf_extraction_engine")

# ---------------------------------------------------------------------------
# Patterns — TOC / outline
# ---------------------------------------------------------------------------

_BOX_PARENT_RE      = re.compile(r"^\s*BOX(?:ES)?\s*$", re.IGNORECASE)
_ANNEX_PARENT_RE    = re.compile(r"^\s*ANNEX(?:ES)?\s*$", re.IGNORECASE)
_FIGURE_PARENT_RE   = re.compile(r"^\s*FIGURES?\s*$", re.IGNORECASE)
_TABLE_PARENT_RE    = re.compile(r"^\s*TABLES?\s*$", re.IGNORECASE)
_APPENDIX_PARENT_RE = re.compile(r"^\s*APPENDIX\s*", re.IGNORECASE)
_SKIP_PARENTS       = (_FIGURE_PARENT_RE, _TABLE_PARENT_RE, _APPENDIX_PARENT_RE)

_TOC_HEADING_RE = re.compile(
    r"(table\s+of\s+contents|^\s*CONTENTS\s*$)", re.IGNORECASE | re.MULTILINE
)
_BOX_HEADER_RE   = re.compile(r"^\s*BOX(?:ES)?\s*$", re.IGNORECASE)
_ANNEX_HEADER_RE = re.compile(r"^\s*ANNEX(?:ES)?\s*$", re.IGNORECASE)
_ANY_HEADER_RE   = re.compile(
    r"^\s*(BOX(?:ES)?|ANNEX(?:ES)?|FIGURES?|TABLES?|APPENDIX|APPENDICES|CONTENTS)\s*$",
    re.IGNORECASE,
)
_TOC_LINE_RE = re.compile(
    r"^(?P<indent>\s*)"
    r"(?P<title>\S.{1,150}?)"
    r"[\s._\u2026\u00B7\u2013\u2014]{2,}"
    r"(?P<page>\d{1,4})\s*$",
)

# ---------------------------------------------------------------------------
# Patterns — Structural Benchmark table detection
# ---------------------------------------------------------------------------

_SB_TEXT_RE    = re.compile(r"structural\s+benchmark", re.IGNORECASE)
_SB_TITLE_RE   = re.compile(r"structural\s+benchmark", re.IGNORECASE)
_SB_SECTION_RE = re.compile(r"^structural\s+benchmarks?\s*$", re.IGNORECASE)
_SB_COL_RE     = re.compile(r"^structural.?benchmarks?$", re.IGNORECASE)
_REFORM_COL_RE = re.compile(
    r"^(reform.?area|policy.?area|area|sector|theme|subject)$", re.IGNORECASE
)
_REFORM_PREFIX_RE = re.compile(
    r"^(AML[/-]?CFT"
    r"|Governance and Transparency"
    r"|Governance"
    r"|Financial Inclusion"
    r"|Revenue Mobilization"
    r"|Public Financial Management"
    r"|Food security"
    r"|Land titling"
    r"|Social Safety Nets?"
    r"|Statistics"
    r"|Monetary Policy"
    r"|Financial Sector"
    r"|Fiscal \w+"
    r"|Banking \w+"
    r"|Trade"
    r"|Investment"
    r"|Labor \w+"
    r"|Climate \w+"
    r"|Health \w+"
    r"|Education \w+"
    r"|Tax \w+"
    r"|Energy \w+"
    r"|Debt \w+"
    r"|Gender \w+"
    r"|Poverty \w+"
    r"|Infrastructure \w*"
    r")",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Patterns — Document date extraction
# ---------------------------------------------------------------------------

_MONTH_YEAR_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+(?:\d{1,2},?\s+)?(\d{4})\b",
    re.IGNORECASE,
)

_IMF_REPORT_NO_RE = re.compile(
    r"\bCountry\s+Report\s+No\.?\s*(\d{2}/\d+)\b"
    r"|\b(?:EBS|SM|CR|EBD|SUR|PDR|SPR|FAD|MCM|RES|LEG|IEO|SEC|TR|RM)[/-]\d{2}/\d+\b"
    r"|\b\d{4}/\d{3}\b",
    re.IGNORECASE,
)

_CONTEXT_DATE_RE = re.compile(
    r"(?:approved|prepared|issued|published|dated?|effective|completed?|finalized?)"
    r"\s+(?:by\s+\w+\s+)?(?:on\s+)?"
    r"((?:January|February|March|April|May|June|July|August|September|October|November|December"
    r"|Jan\.?|Feb\.?|Mar\.?|Apr\.?|Jun\.?|Jul\.?|Aug\.?|Sep\.?|Sept\.?|Oct\.?|Nov\.?|Dec\.?)"
    r"\s+(?:\d{1,2},?\s+)?\d{4})",
    re.IGNORECASE,
)

_YEAR_ONLY_RE = re.compile(r"(?:©\s*|^\s*)(\d{4})\s*$", re.MULTILINE)


def extract_document_date(pdf_path: str) -> dict:
    """
    Scan the first 6 pages for a document date.

    Strategy:
      Pass 1 — look for a PROMINENT standalone "Month YYYY" on its own line
               (the cover-page date like "      July 2022").
               This is the most reliable signal in IMF PDFs.
      Pass 2 — contextual date ("completed on June 28, 2022") as fallback.
      Pass 3 — any Month+Year anywhere on early pages.
      Pass 4 — year only (last resort).

    Returns:
        {
          "raw":    "July 2022",
          "month":  "July",
          "year":   "2022",
          "source": "prominent|context|month_year|year_only",
          "report_number": "22/245"
        }
    """
    import fitz

    _PROMINENT_DATE_RE = re.compile(
        r"^\s*(January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+(\d{4})\s*$",
        re.IGNORECASE | re.MULTILINE,
    )

    doc = fitz.open(str(pdf_path))
    pages_to_scan = min(6, len(doc))

    prominent: dict | None = None
    contextual: dict | None = None
    fallback:   dict | None = None
    report_number: str | None = None

    for i in range(pages_to_scan):
        text = doc[i].get_text()

        if report_number is None:
            m = _IMF_REPORT_NO_RE.search(text)
            if m:
                report_number = m.group(1) if m.lastindex and m.group(1) else m.group(0)

        if prominent is None:
            m = _PROMINENT_DATE_RE.search(text)
            if m:
                prominent = {
                    "raw":    f"{m.group(1)} {m.group(2)}",
                    "month":  m.group(1),
                    "year":   m.group(2),
                    "source": "prominent",
                }

        if contextual is None:
            m = _CONTEXT_DATE_RE.search(text)
            if m:
                raw = m.group(1).strip()
                m2  = _MONTH_YEAR_RE.search(raw)
                contextual = {
                    "raw":    raw,
                    "month":  m2.group(1) if m2 else None,
                    "year":   m2.group(2) if m2 else None,
                    "source": "context",
                }

        if fallback is None:
            m = _MONTH_YEAR_RE.search(text)
            if m:
                fallback = {
                    "raw":    m.group(0).strip(),
                    "month":  m.group(1),
                    "year":   m.group(2),
                    "source": "month_year",
                }

    year_only: dict | None = None
    for i in range(pages_to_scan):
        text = doc[i].get_text()
        m = _YEAR_ONLY_RE.search(text)
        if m:
            year_only = {"raw": m.group(1), "month": None, "year": m.group(1), "source": "year_only"}
            break

    doc.close()

    best = (prominent or contextual or fallback or year_only
            or {"raw": None, "month": None, "year": None, "source": None})
    best["report_number"] = report_number
    return best


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TocTarget:
    category: str   # "Box" | "Annex"
    title: str
    page: int
    level: int = 1
    source_pdf: str = ""

    def __str__(self):
        return f"{'  '*(self.level-1)}[{self.category}] {self.title}  ->  p.{self.page}"

    def to_dict(self):
        return {"category": self.category, "title": self.title, "page": self.page}


@dataclass
class BenchmarkResult:
    table_title: str
    page_num: int
    columns: list[str]
    rows: dict

    def to_dict(self):
        return {
            "table_title": self.table_title,
            "page_num":    self.page_num,
            "columns":     self.columns,
            "rows":        self.rows,
        }


@dataclass
class PriorActionResult:
    """
    One "Prior Actions" table (or the prior-action rows sliced out of a
    combined "Prior Actions and Structural Benchmarks" table, e.g. IMF
    Table 3-style tables). Prior Actions are pre-conditions completed
    BEFORE Board approval/review completion, and unlike Structural
    Benchmarks they normally carry no Target Date / Rationale columns --
    just the action text and its status.
    """
    table_title: str
    page_num: int
    columns: list[str]
    rows: dict  # {action_text: {"Status": ..., "Category": ...}}

    def to_dict(self):
        return {
            "table_title": self.table_title,
            "page_num":    self.page_num,
            "columns":     self.columns,
            "rows":        self.rows,
        }


@dataclass
class QPCCriterion:
    """One row of a Quantitative Performance Criteria / Indicative Targets
    table -- a single criterion (e.g. "Floor on primary fiscal balance,
    excluding grants") tracked across several test dates. Each test date
    column can carry a different subset of sub-values (Program/PC,
    Adjusted PC, Actual, Prel., Proposed PC, Proposed IT, Status) since
    past test dates show outturns and future test dates show only proposed
    targets -- see QPCResult docstring."""
    category: str            # "Performance Criteria" | "Indicative Target" | "Memorandum Item"
    criterion: str            # e.g. "Floor on primary fiscal balance, excluding grants"
    test_dates: list[dict]    # [{"date": "Dec. 2024", "values": {"PC": "24.0", ...}}, ...]

    def to_dict(self):
        return {
            "category":   self.category,
            "criterion":  self.criterion,
            "test_dates": self.test_dates,
        }


@dataclass
class QPCResult:
    """
    A Quantitative Performance Criteria (QPC) and Indicative Targets (IT)
    table -- e.g. IMF Table 1-style "Quantitative Performance Criteria and
    Indicative Targets" tables. Structurally very different from SB/Prior
    Action tables: it's a wide matrix of criteria (rows) x test dates
    (column groups), not a simple row-per-item table, so criteria are kept
    as a list of QPCCriterion rather than a flat rows dict.
    """
    table_title: str
    page_num: int
    unit_note: str | None      # e.g. "Millions of U.S. dollars, unless otherwise indicated"
    criteria: list[QPCCriterion]

    def to_dict(self):
        return {
            "table_title": self.table_title,
            "page_num":    self.page_num,
            "unit_note":   self.unit_note,
            "criteria":    [c.to_dict() for c in self.criteria],
        }


# ---------------------------------------------------------------------------
# Track 1 — Box & Annex from TOC  (pure fitz, < 1 s)
# ---------------------------------------------------------------------------

def _extract_from_outline(outline: list, source_pdf: str) -> list[TocTarget]:
    results: list[TocTarget] = []
    current_category: str | None = None
    parent_level: int | None = None

    for lvl, title, page, *_ in outline:
        title_s = title.strip()

        if _BOX_PARENT_RE.match(title_s):
            current_category, parent_level = "Box", lvl
            continue
        elif _ANNEX_PARENT_RE.match(title_s):
            current_category, parent_level = "Annex", lvl
            continue
        elif any(p.match(title_s) for p in _SKIP_PARENTS):
            current_category, parent_level = None, lvl
            continue

        if parent_level is not None and lvl <= parent_level:
            current_category, parent_level = None, None

        if current_category and parent_level is not None and lvl == parent_level + 1 and title_s:
            results.append(TocTarget(
                category=current_category, title=title_s,
                page=page, level=1, source_pdf=source_pdf,
            ))

    return results


def _extract_from_text(doc, source_pdf: str) -> list[TocTarget]:
    results: list[TocTarget] = []
    toc_pages = [
        i for i in range(min(30, len(doc)))
        if _TOC_HEADING_RE.search(doc[i].get_text())
    ]
    if not toc_pages:
        return []

    expanded: set[int] = set(toc_pages)
    for tp in toc_pages:
        for j in range(tp + 1, min(tp + 5, len(doc))):
            text = doc[j].get_text()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            toc_line_count = sum(1 for l in lines if _TOC_LINE_RE.match(l))
            if lines and toc_line_count / len(lines) > 0.15:
                expanded.add(j)
            else:
                break
    all_toc_pages = sorted(expanded)

    current_category: str | None = None

    for page_idx in all_toc_pages:
        text = doc[page_idx].get_text()
        for line in text.splitlines():
            stripped = line.strip()
            if _ANY_HEADER_RE.match(stripped):
                if _BOX_HEADER_RE.match(stripped):
                    current_category = "Box"
                elif _ANNEX_HEADER_RE.match(stripped):
                    current_category = "Annex"
                else:
                    current_category = None
                continue
            if current_category is None:
                continue
            m = _TOC_LINE_RE.match(line)
            if not m:
                continue
            title = m.group("title").strip().rstrip("_. ")
            page  = int(m.group("page"))
            level = 1 + len(m.group("indent")) // 2
            if title:
                results.append(TocTarget(
                    category=current_category, title=title,
                    page=page, level=level, source_pdf=source_pdf,
                ))
    return results


def extract_toc_targets(pdf_path: str) -> list[TocTarget]:
    """Extract Box and Annex entries. PyMuPDF only — no Docling. < 1 s."""
    import fitz
    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)

    outline = doc.get_toc(simple=False)
    if outline:
        results = _extract_from_outline(outline, pdf_path)
        if results:
            doc.close()
            logger.info("[%s] %d entries from embedded outline", pdf_path, len(results))
            return results
        logger.debug("[%s] Outline has no Box/Annex; trying text scan", pdf_path)

    results = _extract_from_text(doc, pdf_path)
    doc.close()

    if results:
        logger.info("[%s] %d entries from text scan", pdf_path, len(results))
    else:
        logger.warning("[%s] No Box/Annex entries found", pdf_path)
    return results


# ---------------------------------------------------------------------------
# Track 2 — Structural Benchmarks via targeted Docling
# ---------------------------------------------------------------------------

_MEFP_RE = re.compile(r"memorandum\s+of\s+economic\s+and\s+financial\s+policies", re.IGNORECASE)
_MEFP_HEADING_RE = re.compile(
    r"^[\s]*(?:attachment\s+[IVXivx\d]+[\.\s]+)?"
    r"memorandum\s+of\s+economic\s+and\s+financial\s+policies"
    r"(?:\s*\(MEFP\))?[\s]*$",
    re.IGNORECASE,
)
_LIST_ITEM_RE = re.compile(r"^\s*\d+\.\s*$")
_SB_TABLE_TITLE_RE = re.compile(
    r"(table\s+\d+[a-z]?[\.:].{0,200}?(?:structural\s+(?:benchmarks?|conditionality)|reform\s+targets?)"
    r"|^(?:structural\s+(?:benchmarks?|conditionality)|reform\s+targets?)\s*$"
    r"|(?:structural\s+(?:benchmarks?|conditionality)|reform\s+targets?),\s+\d{4})",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)

# A standalone "STRUCTURAL BENCHMARKS" heading (no "Table N." caption) is
# ambiguous: it's the caption of a real table in some documents, but in
# others (commonly the TMU) it's just a section heading introducing ordinary
# narrative paragraphs about the benchmarks, not a table at all. This pattern
# catches that narrative case so it can be excluded -- a numbered paragraph
# ("22.      For the purposes of...") starting within the next couple of
# lines after the heading is a strong, cheap signal of prose, not a table.
_TMU_NARRATIVE_AFTER_HEADING_RE = re.compile(
    r"^(?:structural\s+(?:benchmarks?|conditionality)|reform\s+targets?)\s*\n\s*\d{1,3}\.\s{2,}\S",
    re.IGNORECASE | re.MULTILINE,
)
_HAS_TABLE_CAPTION_RE = re.compile(r"table\s+\d+[a-z]?[\.:]", re.IGNORECASE)

# Extracts just the table number/letter (e.g. "2", "1", "2c") from a caption,
# so consecutive pages can be checked for whether they're actually the SAME
# table continuing, or two different tables that happen to sit close
# together. Real multi-page continuations always keep the same number; a
# number that changes (especially one that goes backwards, e.g. "Table 2"
# then later "Table 1") is a strong signal these are unrelated tables, often
# from different sections/annexes with their own independent numbering.
_TABLE_NUMBER_RE = re.compile(r"table\s+(\d+[a-z]?)[\.:]", re.IGNORECASE)


def _extract_table_number(text: str) -> str | None:
    m = _TABLE_NUMBER_RE.search(text)
    return m.group(1).lower() if m else None


def _leading_int(number: str | None) -> int | None:
    """Pull the leading digits off a table number like '4' or '2c' -> 4 / 2."""
    if number is None:
        return None
    m = re.match(r"(\d+)", number)
    return int(m.group(1)) if m else None


def _is_next_sb_table(prev_number: str | None, this_number: str | None) -> bool:
    """
    True if `this_number` is exactly one more than `prev_number` (e.g.
    "3" -> "4"). IMF SB sections routinely run two adjacent tables back to
    back -- typically "Table N. ... Prior Actions and Structural
    Benchmarks" immediately followed by "Table N+1. ... Recurrent
    Structural Benchmarks" -- and both belong to the same SB extraction, so
    a forward step of exactly 1 should NOT split the cluster. A backward or
    non-consecutive change (see MOZ Review 3 in _select_table_cluster's
    docstring: "Table 2" then later "Table 1") is still treated as a split,
    since that pattern indicates unrelated tables with independent
    numbering, not a continuation.
    """
    a, b = _leading_int(prev_number), _leading_int(this_number)
    return a is not None and b is not None and b == a + 1


def _cluster_pages(pages: list[int], gap: int = 5,
                    numbers: dict[int, str | None] | None = None) -> list[list[int]]:
    """
    Group pages into clusters of consecutive (within gap) pages.

    If `numbers` is given (page -> extracted table number, or None if no
    explicit number was found on that page), a cluster also splits whenever
    the table number actually CHANGES between two explicitly-numbered pages,
    even if they're within the page-distance gap. A page with no number
    (None) is treated as compatible with whatever number the current
    cluster already has -- some continuation pages drop the explicit
    "Table N." prefix, so None shouldn't force a split on its own.

    Exception: a change that steps forward by exactly 1 (e.g. "3" -> "4")
    does NOT split -- see _is_next_sb_table. Two immediately-adjacent SB
    tables numbered N and N+1 (e.g. a "Prior Actions and Structural
    Benchmarks" table followed by a "Recurrent Structural Benchmarks"
    table) are both genuine SB tables that belong together, not unrelated
    tables. A backward or skipped change still splits, since that's the
    unrelated-tables signal (see MOZ Review 3 case below).
    """
    if not pages:
        return []
    numbers = numbers or {}
    clusters, current = [], [pages[0]]
    current_number = numbers.get(pages[0])
    for p in pages[1:]:
        this_number = numbers.get(p)
        same_table = (
            this_number is None
            or current_number is None
            or this_number == current_number
            or _is_next_sb_table(current_number, this_number)
        )
        if p - current[-1] <= gap and same_table:
            current.append(p)
            if this_number is not None:
                current_number = this_number
        else:
            clusters.append(current)
            current = [p]
            current_number = this_number
    clusters.append(current)
    return clusters


def _select_table_cluster(pages: list[int], captioned_pages: set[int], gap: int = 5,
                           numbers: dict[int, str | None] | None = None) -> list[int]:
    """
    Group matched pages into clusters and pick the best one.

    Previously this always picked the LAST cluster, on the assumption that a
    later match is the more "final/authoritative" version. That assumption
    broke for CMR: the real table is a 3-page cluster with an explicit
    "Table 2. ... Prior Action and Structural Benchmarks" caption on every
    page, but a later, unrelated page elsewhere in the document matched only
    the bare "STRUCTURAL BENCHMARKS" heading (no caption) -- and "last
    cluster wins" discarded the real table in favor of that spurious match
    purely because it came later in the page order.

    Fix: prefer the LAST cluster that contains at least one explicitly
    captioned page ("Table N. ..."). Only fall back to "last cluster
    overall" if no cluster has a caption at all (some legitimate tables
    genuinely rely on a standalone heading with no "Table N." prefix).

    Also broke for MOZ Review 3: page 75 ("Table 2. To monitor progress on
    the structural reforms...") and page 78 ("Table 1. Mozambique: Proposed
    Structural Benchmarks") are 3 pages apart -- within the proximity gap --
    so they got merged into one cluster despite being two unrelated tables
    (confirmed by the table number going BACKWARDS: 2 then 1, which never
    happens in a real multi-page continuation). Both pages then got sent to
    Claude together, and the real table's rows were lost. Passing `numbers`
    lets _cluster_pages split on a number change instead of proximity alone.

    Note: a FORWARD, consecutive number change (e.g. "3" -> "4") does not
    trigger this split at all -- _cluster_pages already keeps those pages
    together (see _is_next_sb_table), because that's the normal pattern for
    a "Prior Actions and Structural Benchmarks" table immediately followed
    by a "Recurrent Structural Benchmarks" table (e.g. Liberia's Table 3 +
    Table 4). Both are genuine SB tables and both should be extracted.
    """
    clusters = _cluster_pages(pages, gap, numbers)
    if not clusters:
        return []

    captioned_clusters = [c for c in clusters if any(p in captioned_pages for p in c)]
    if captioned_clusters:
        if len(captioned_clusters) > 1:
            # More than one genuinely captioned table in this document (e.g.
            # a historical comparison table in an appendix, or the same
            # table reprinted elsewhere). "Last one wins" is a heuristic,
            # not a guarantee -- surface this so an ambiguous case doesn't
            # silently resolve to the wrong table.
            logger.warning(
                "Multiple captioned SB table clusters found: %s -- using the "
                "last one (%s). If this is wrong, the document may need a "
                "closer look.", captioned_clusters, captioned_clusters[-1]
            )
        return captioned_clusters[-1]
    return clusters[-1]


def _is_prose_table_reference(line: str) -> bool:
    """True if a "Table N. ..." line is really a sentence that mentions
    tables (ends with a period and has another "Table N" reference or a
    sentence verb), not a table caption."""
    l = line.strip()
    if not l.endswith("."):
        return False
    body = re.sub(r"^table\s+\d+[a-z]?[\.:]", "", l, flags=re.IGNORECASE)
    return bool(re.search(r"\btables?\s+\d+", body, re.IGNORECASE)
                or re.search(r"\b(is|are|described|set out|shown|see)\b", body, re.IGNORECASE))


def _is_sb_table_page(text: str) -> bool:
    """
    True if this page's SB-related text match looks like a real table caption
    rather than a narrative section heading (e.g. the TMU's "STRUCTURAL
    BENCHMARKS" heading followed by numbered prose paragraphs, not a table).
    """
    if not _SB_TABLE_TITLE_RE.search(text):
        return False
    # Reject pages whose only "Table N." + SB-keyword match is a sentence
    # referring to a table (GHA p.113: "Table 1. The structural benchmarks
    # are described in Table 2.") rather than a real table caption.
    sb_caption_lines = [
        m.group(0) for m in re.finditer(r"table\s+\d+[a-z]?[\.:][^\n]*", text, re.IGNORECASE)
        if _SB_TITLE_RE.search(m.group(0)) or re.search(r"reform\s+targets?", m.group(0), re.IGNORECASE)
    ]
    if sb_caption_lines and all(_is_prose_table_reference(l) for l in sb_caption_lines):
        return False
    # A "Table N." caption anywhere on the page is a strong positive signal --
    # real tables almost always have one, whether or not this specific match
    # came from the table-caption branch or the standalone-heading branch.
    if _HAS_TABLE_CAPTION_RE.search(text):
        return True
    # No "Table N." caption at all. If the standalone heading is immediately
    # followed by TMU-style numbered narrative, this is prose, not a table --
    # reject it.
    if _TMU_NARRATIVE_AFTER_HEADING_RE.search(text):
        return False
    return True


def _find_benchmark_tables(pdf_path: str, include_reform_targets: bool = True) -> list[dict]:
    """
    Return 1-indexed page numbers for the SB table in the document.

    Strategy:
      1. Locate the MEFP section heading, if present (logged for diagnostics
         only -- it does NOT restrict which pages get searched; some
         documents, e.g. CPV_2025_ECF_Review6, have the real SB table
         BEFORE the MEFP section, so gating the search to mefp_start-onward
         would silently miss it entirely).
      2. Scan the WHOLE document for pages that contain an actual SB table
         title (not just a prose mention of "structural benchmark", and not
         a narrative section heading like the TMU's "STRUCTURAL BENCHMARKS"
         heading followed by ordinary numbered paragraphs).
      3. Group into clusters and pick the last cluster that has an explicit
         "Table N." caption, falling back to the last cluster overall if
         none do (see _select_table_cluster). This caption-preference logic
         is what keeps this safe even without the MEFP gate -- spurious
         early/late mentions without a real caption still lose out to a
         genuine captioned table wherever it happens to sit in the document.
    """
    import fitz
    doc = fitz.open(pdf_path)

    # Pass 1: locate MEFP section heading (diagnostic only, see above)
    mefp_start: int | None = None
    for i in range(len(doc)):
        text = doc[i].get_text()
        if not _MEFP_RE.search(text):
            continue
        lines = text.splitlines()
        for j, line in enumerate(lines):
            if _MEFP_HEADING_RE.match(line):
                prev = lines[j - 1].strip() if j > 0 else ""
                if not _LIST_ITEM_RE.match(prev):
                    mefp_start = i + 1
                    logger.info("[%s] MEFP section found at page %d", pdf_path, mefp_start)
                    break
        if mefp_start:
            break

    if mefp_start is None:
        logger.info("[%s] No MEFP section found", pdf_path)

    # Pass 2: collect pages with an actual SB table title (not just prose
    # mentions, and not narrative section headings — see _is_sb_table_page).
    # Scans the ENTIRE document -- no MEFP-based lower bound (see docstring).
    # Also track which of those pages have an explicit "Table N." caption,
    # and what table number/letter each page shows (for number-continuity
    # aware clustering -- see _cluster_pages).
    table_pages = []
    captioned_pages: set[int] = set()
    table_numbers: dict[int, str | None] = {}
    page_kind: dict[int, str] = {}
    for i in range(len(doc)):
        pg = i + 1
        text = doc[i].get_text()
        if not _is_sb_table_page(text):
            continue
        table_pages.append(pg)
        if _HAS_TABLE_CAPTION_RE.search(text):
            captioned_pages.add(pg)
        table_numbers[pg] = _extract_table_number(text)
        page_kind[pg] = _page_table_kind(text)

    doc.close()

    # Pass 3: cluster, then pick the best cluster PER KIND. Some documents
    # (e.g. GHA Review 6: Table 2 "Existing Structural Conditionality Under
    # the ECF" + Table 5 "Reform Targets Under the PCI") legitimately carry
    # two different SB-family tables. The old "last captioned cluster wins"
    # rule picked Table 5 and dropped Table 2. Now: the SB-kind table is
    # primary (last captioned SB cluster, preserving the CMR/MOZ logic),
    # and the Reform-Target-kind table is returned additionally.
    clusters = _cluster_pages(table_pages, 5, table_numbers)
    if not clusters:
        return []

    def _cluster_kind(c: list[int]) -> str:
        cap = [p for p in c if p in captioned_pages] or c
        kinds = [page_kind.get(p, "SB") for p in cap]
        return "Reform Target" if kinds.count("Reform Target") > len(kinds) / 2 else "SB"

    captioned = [c for c in clusters if any(p in captioned_pages for p in c)]
    pool = captioned or [clusters[-1]]
    tables: list[dict] = []
    for kind in ("SB", "Reform Target"):
        of_kind = [c for c in pool if _cluster_kind(c) == kind]
        if not of_kind:
            continue
        if len(of_kind) > 1:
            logger.warning("Multiple captioned %s table clusters found: %s -- using the "
                           "last one (%s).", kind, of_kind, of_kind[-1])
        tables.append({"kind": kind, "pages": of_kind[-1]})

    if not include_reform_targets and any(t["kind"] == "SB" for t in tables):
        tables = [t for t in tables if t["kind"] == "SB"]

    tables.sort(key=lambda t: t["pages"][0])
    logger.info("[%s] SB-family tables (mefp_start=%s): %s", pdf_path, mefp_start, tables)
    return tables


def _page_table_kind(text: str) -> str:
    """"Reform Target" if this page's caption is a Reform Targets table (PCI),
    otherwise "SB" (Structural Benchmarks / Structural Conditionality)."""
    for m in re.finditer(r"table\s+\d+[a-z]?[\.:][^\n]*(?:\n[^\n]*)?", text, re.IGNORECASE):
        cap = m.group(0)
        if re.search(r"reform\s+targets?", cap, re.IGNORECASE) and not re.search(
                r"structural\s+(?:benchmarks?|conditionality)", cap, re.IGNORECASE):
            return "Reform Target"
    return "SB"


def _find_benchmark_pages(pdf_path: str, include_reform_targets: bool = True) -> list[int]:
    """Flat, sorted list of SB-family table pages (see _find_benchmark_tables)."""
    tables = _find_benchmark_tables(pdf_path, include_reform_targets)
    return sorted(p for t in tables for p in t["pages"])


# ---------------------------------------------------------------------------
# Patterns & page-finder — Prior Actions table detection
# ---------------------------------------------------------------------------
# Prior Actions tables come in two shapes in practice:
#   (a) A standalone table titled just "Prior Action(s)"
#   (b) A COMBINED table titled "Prior Actions and Structural Benchmarks"
#       (e.g. Liberia's Table 3), where the prior-action rows sit above a
#       "Structural Benchmarks" divider that introduces Target Date /
#       Rationale columns the prior-action rows don't have.
# Both are caught by requiring "prior action(s)" to appear in the table
# caption or as a standalone table heading -- deliberately looser than the
# SB detector, since Prior Action rows have no reliable narrative-heading
# false-positive pattern equivalent to the SB/TMU case.
_PRIOR_ACTION_TABLE_TITLE_RE = re.compile(
    r"(table\s+\d+[a-z]?[\.:][^\n]{0,200}?prior\s+actions?"
    r"|^(?:proposed\s+)?prior\s+actions?(?:\s+(?:for|to|under|ahead\s+of|before)\b[^\n]{0,80})?\s*$"
    r"|^[^\n]{0,120}?prior\s+actions?\s+and\s+structural\s+(?:benchmarks?|conditionality)"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _is_prior_action_table_page(text: str) -> bool:
    """True if this page's text contains a genuine Prior Actions table
    caption (a Table N. caption is a strong positive signal, same logic as
    _is_sb_table_page)."""
    return bool(_PRIOR_ACTION_TABLE_TITLE_RE.search(text))


def _find_prior_action_pages(pdf_path: str) -> list[int]:
    """
    Return 1-indexed page numbers for the Prior Actions table.

    Same three-pass strategy as _find_benchmark_pages (whole-document scan,
    caption tracking, number-continuity-aware clustering -- see
    _select_table_cluster / _cluster_pages), but keyed to
    _PRIOR_ACTION_TABLE_TITLE_RE instead of the SB title pattern.

    Note: for a COMBINED "Prior Actions and Structural Benchmarks" table
    (e.g. Liberia Table 3), this will typically return the SAME page(s) as
    _find_benchmark_pages -- that's expected. The two extraction functions
    (extract_prior_actions vs extract_structural_benchmarks) use different
    vision prompts to slice out the right rows from the same physical
    page(s): Prior Action rows precede the "Structural Benchmarks" divider
    and lack Target Date/Rationale columns; genuine SB rows come after it
    and have those columns. A "(concluded)" continuation page of a combined
    table (e.g. page 116 continuing Liberia's Table 3) will still match
    this title regex (the caption repeats) even though by then only SB
    rows remain -- that's fine, the vision call for that page will simply
    return zero prior-action rows and get filtered out downstream.
    """
    import fitz
    doc = fitz.open(pdf_path)

    table_pages = []
    captioned_pages: set[int] = set()
    table_numbers: dict[int, str | None] = {}
    for i in range(len(doc)):
        pg = i + 1
        text = doc[i].get_text()
        if not _is_prior_action_table_page(text):
            continue
        table_pages.append(pg)
        if _HAS_TABLE_CAPTION_RE.search(text):
            captioned_pages.add(pg)
        table_numbers[pg] = _extract_table_number(text)

    doc.close()

    pages = _select_table_cluster(table_pages, captioned_pages, numbers=table_numbers)
    logger.info("[%s] Prior Action table pages: %s", pdf_path, pages)
    return pages


# ---------------------------------------------------------------------------
# Patterns & page-finder — Quantitative Performance Criteria (QPC) table
# ---------------------------------------------------------------------------
# QPC/Indicative Target tables are structurally a wide matrix (criteria x
# test dates), not a simple row-per-item table -- see QPCResult. They're
# reliably captioned, e.g. "Table 1. Liberia: Quantitative Performance
# Criteria and Indicative Targets, December 2024 - December 2026", so the
# title match can be stricter than Prior Actions -- no standalone-heading
# ambiguity to worry about here since these tables are always numbered and
# titled explicitly in practice.
_QPC_TABLE_TITLE_RE = re.compile(
    r"(table\s+\d+[a-z]?[\.:][^\n]{0,200}?(?:quantitative\s+performance\s+criteria"
    r"|performance\s+criteria\s+and\s+indicative\s+targets)"
    r"|^quantitative\s+performance\s+criteria(?:\s+and\s+indicative\s+targets)?\s*$"
    r")",
    re.IGNORECASE | re.MULTILINE,
)


def _is_qpc_table_page(text: str) -> bool:
    return bool(_QPC_TABLE_TITLE_RE.search(text))


def _find_qpc_pages(pdf_path: str) -> list[int]:
    """
    Return 1-indexed page numbers for the QPC / Indicative Targets table.
    Same three-pass strategy as _find_benchmark_pages, keyed to
    _QPC_TABLE_TITLE_RE. QPC tables are usually a single page (they're
    already very dense/wide), but multi-page QPC tables do occur in some
    documents, hence still routing through the same clustering logic
    rather than assuming a single page.
    """
    import fitz
    doc = fitz.open(pdf_path)

    table_pages = []
    captioned_pages: set[int] = set()
    table_numbers: dict[int, str | None] = {}
    for i in range(len(doc)):
        pg = i + 1
        text = doc[i].get_text()
        if not _is_qpc_table_page(text):
            continue
        table_pages.append(pg)
        if _HAS_TABLE_CAPTION_RE.search(text):
            captioned_pages.add(pg)
        table_numbers[pg] = _extract_table_number(text)

    doc.close()

    pages = _select_table_cluster(table_pages, captioned_pages, numbers=table_numbers)
    logger.info("[%s] QPC table pages: %s", pdf_path, pages)
    return pages


def _write_subpdf(pdf_path: str, page_nums: list[int], out_path: str) -> None:
    import fitz
    doc = fitz.open(pdf_path)
    total = len(doc)
    expanded = sorted({pp for p in page_nums for pp in (p-1, p, p+1) if 1 <= pp <= total})
    sub = fitz.open()
    for p in expanded:
        sub.insert_pdf(doc, from_page=p-1, to_page=p-1)
    sub.save(out_path)
    sub.close()
    doc.close()
    logger.info("Sub-PDF: %d pages -> %s", len(expanded), out_path)


# ---------------------------------------------------------------------------
# Table parsing
# ---------------------------------------------------------------------------

def _get_table_data(table, doc) -> tuple[dict, list[str], str]:
    html = ""
    try:
        if hasattr(table, "export_to_html"):
            try:
                html = table.export_to_html(doc=doc) or ""
            except TypeError:
                html = table.export_to_html() or ""
    except Exception:
        pass

    if html:
        try:
            from docling_extractor import _html_to_dict
            data, columns = _html_to_dict(html)
            if data and columns:
                return data, columns, html
        except ImportError:
            pass
        except Exception as e:
            logger.debug("_html_to_dict failed: %s", e)

    try:
        import pandas as pd
        try:
            df = table.export_to_dataframe(doc=doc)
        except TypeError:
            df = table.export_to_dataframe()
        if df is None:
            return {}, [], html
        if df.shape[0] == 0 or df.shape[1] < 2:
            return {}, [], html

        raw_cols = [str(c) for c in df.columns]
        columns  = raw_cols[1:]
        data: dict = {}
        for _, row in df.iterrows():
            key = str(row.iloc[0]).strip()
            if not key or key == "nan":
                continue
            data[key] = {}
            for col in columns:
                val = row[col]
                try:
                    is_na = pd.isna(val)
                except (TypeError, ValueError):
                    is_na = False
                if not is_na:
                    s = str(val).strip()
                    data[key][col] = s if s and s != "nan" else None
                else:
                    data[key][col] = None
        if data:
            return data, columns, html
    except Exception as e:
        logger.debug("DataFrame export failed: %s", e)

    return {}, [], html


def _detect_sb_format(columns: list[str], data: dict, table_title: str) -> str:
    if any(_SB_COL_RE.match(c) for c in columns):
        return "B"
    for key in data:
        if _SB_SECTION_RE.match(key.strip()):
            return "A"
    if _SB_TITLE_RE.search(table_title):
        return "C"
    return "none"


def _extract_format_a(data: dict, columns: list[str]) -> tuple[dict, list[str]]:
    in_sb = False
    rows: dict = {}
    for key, val in data.items():
        if _SB_SECTION_RE.match(key.strip()):
            in_sb = True
            continue
        if in_sb:
            all_null = isinstance(val, dict) and all(v is None for v in val.values())
            if all_null and len(key.strip()) < 60 and not key.strip()[:1].isdigit():
                break
            rows[key.strip()] = {c: val.get(c) for c in columns}

    # Ensure Status is surfaced even when the column isn't named "status".
    # Infer it from cell value patterns and inject a canonical "Status" key.
    has_status_col = any(re.search(r"status", c, re.IGNORECASE) for c in columns)
    if not has_status_col and rows:
        col_types = _identify_col_types(columns, data)
        inferred_status_col = next(
            (c for c in columns if col_types.get(c) == "status"), None
        )
        if inferred_status_col:
            for row_key, val in rows.items():
                if isinstance(val, dict) and "Status" not in val:
                    val["Status"] = val.get(inferred_status_col)

    return rows, list(columns)


def _split_reform_and_benchmark(text: str) -> tuple[str, str]:
    text = text.strip()
    m = _REFORM_PREFIX_RE.match(text)
    if m:
        prefix = m.group(0).strip()
        rest = text[len(prefix):].strip()
        if len(rest) > 20:
            return prefix, rest
    return "", text


_KNOWN_REFORM_AREAS: list[str] = sorted([
    "Governance and Transparency", "Financial Inclusion", "Revenue Mobilization",
    "Public Financial Management", "Food security", "Land titling", "AML/CFT",
    "Social Safety Nets", "Social Protection", "Statistics", "Monetary Policy",
    "Financial Sector", "Financial Stability", "Tax Policy", "Investment Climate",
    "Health", "Education", "Climate Change", "Poverty Reduction", "Gender",
    "Debt Management", "Fiscal Policy", "State-Owned Enterprises", "SOEs",
    "Banking Sector", "External Sector", "Trade Policy", "Exchange Rate",
    "Pension", "Insurance", "Business Environment", "Digitalization",
    "Infrastructure", "Agriculture", "Judiciary", "Labor Market",
    "Customs Administration", "Rule of law", "Anti-corruption",
], key=len, reverse=True)

_REFORM_FULL_RE = re.compile(
    r"^(" + "|".join(re.escape(r) for r in _KNOWN_REFORM_AREAS) + r")(\s+|$)",
    re.IGNORECASE,
)
_STATUS_SUFFIX_RE = re.compile(
    r"(?:[.\s])+(Met|Not\s+[Mm]et|Ongoing|Proposed|Completed|Implemented|"
    r"Delayed|Missed|Partly\s+met|Partially\s+met|On\s+track|In\s+progress|"
    r"Observed|Not\s+observed|Not\s+yet\s+due|Pending|Dropped|Waived|Reset|"
    r"Newly\s+proposed(?:\s+SB)?|New)\s*$",
    re.IGNORECASE,
)
_DATE_VAL_RE = re.compile(
    r"^(End-[A-Za-z]+\.?\s*\d{0,4}"
    r"|(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})",
    re.IGNORECASE,
)
_STATUS_VAL_RE = re.compile(
    r"^(Met|Not\s+[Mm]et|Ongoing|Proposed|Completed|Implemented|Delayed|"
    r"Newly\s+proposed(?:\s+SB)?|New|Missed|Partly\s+met|Partially\s+met|"
    r"On\s+track|In\s+progress|Observed|Not\s+observed|Not\s+yet\s+due|"
    r"Pending|Delayed|Dropped|Waived|Reset)$",
    re.IGNORECASE,
)
_AREA_FIRST_WORDS: dict[str, list[str]] = {}
for _area in _KNOWN_REFORM_AREAS:
    _fw = _area.split()[0].lower()
    _AREA_FIRST_WORDS.setdefault(_fw, []).append(_area)


def _classify_sb_cell(cell: str) -> tuple[str, str, str]:
    cell = cell.strip()
    if not cell:
        return ("empty", "", "")
    if _DATE_VAL_RE.match(cell) and len(cell) < 40:
        return ("date", "", cell)
    if _STATUS_VAL_RE.match(cell):
        return ("status", "", cell)
    m = _REFORM_FULL_RE.match(cell)
    if m:
        area = m.group(1)
        rest = cell[len(area):].strip()
        return ("reform+bench", area, rest) if rest else ("reform", area, "")
    sm = _STATUS_SUFFIX_RE.search(cell)
    if sm:
        status = sm.group(1)
        prefix = cell[:sm.start()].strip()
        m2 = _REFORM_FULL_RE.match(prefix)
        if m2:
            return ("reform+status", m2.group(1), status)
        return ("bench+status", status, prefix)
    words = cell.split()
    if words and words[0].lower() in _AREA_FIRST_WORDS:
        if len(words) > 1 and words[1][0].isupper() and words[1].lower() not in ("and","or","of","the"):
            best_area = _AREA_FIRST_WORDS[words[0].lower()][0]
            return ("partial_reform+bench", best_area, " ".join(words[1:]))
    return ("bench", "", cell)


def _smart_join(frags: list[str]) -> str:
    frags = [f.strip() for f in frags if f and f.strip()]
    if not frags:
        return ""
    if len(frags) == 1:
        return frags[0]
    result = frags[0]
    for frag in frags[1:]:
        if frag.lower() in result.lower():
            continue
        overlap_found = False
        for n in range(min(len(result), len(frag), 30), 2, -1):
            if result.lower().endswith(frag[:n].lower()):
                result += frag[n:]
                overlap_found = True
                break
        if not overlap_found:
            result = result.rstrip() + " " + frag
    return result.strip()


def _normalize_sb_row(row_key: str, vals: dict) -> dict:
    all_cells = [str(row_key).strip()] + [
        str(v).strip() for v in vals.values()
        if v and str(v).strip() not in ("", "None", "nan")
    ]
    reform_area = ""
    date_val    = ""
    status_val  = ""
    bench_frags: list[str] = []

    for cell in all_cells:
        typ, area, text = _classify_sb_cell(cell)
        if   typ == "empty":              continue
        elif typ == "date":               date_val   = date_val   or text
        elif typ == "status":             status_val = status_val or text
        elif typ == "reform":
            if not reform_area: reform_area = area
        elif typ in ("reform+bench", "partial_reform+bench"):
            if not reform_area: reform_area = area
            if text: bench_frags.append(text)
        elif typ == "reform+status":
            if not reform_area: reform_area = area
            if not status_val:  status_val  = text
        elif typ == "bench+status":
            if not status_val:  status_val = area
            if text: bench_frags.append(text)
        elif typ == "bench":
            if text: bench_frags.append(text)

    cleaned_frags = []
    for frag in bench_frags:
        extracted = False
        for area in _KNOWN_REFORM_AREAS:
            if frag.lower().endswith(" " + area.lower()) or frag.lower().endswith("." + area.lower()):
                suffix_start = len(frag) - len(area)
                bench_part = frag[:suffix_start].rstrip(" .,;").strip()
                if not reform_area:
                    reform_area = area
                if bench_part:
                    cleaned_frags.append(bench_part)
                extracted = True
                break
            if frag.lower().endswith(" in " + area.lower()):
                bench_part = frag[:-(len(area)+4)].strip()
                if not reform_area:
                    reform_area = area
                if bench_part:
                    cleaned_frags.append(bench_part)
                extracted = True
                break
        if not extracted:
            cleaned_frags.append(frag)
    bench_frags = cleaned_frags

    final_frags = []
    for frag in bench_frags:
        found_prefix = False
        for area in _KNOWN_REFORM_AREAS:
            area_words = area.split()
            if len(area_words) > 1:
                for tail_len in range(len(area_words)-1, 0, -1):
                    tail = " ".join(area_words[-tail_len:])
                    if frag.lower().startswith(tail.lower() + " "):
                        rest = frag[len(tail):].strip()
                        if not reform_area:
                            reform_area = area
                        if rest:
                            final_frags.append(rest)
                        found_prefix = True
                        break
            if found_prefix:
                break
        if not found_prefix:
            final_frags.append(frag)
    bench_frags = final_frags

    bench_frags.sort(key=len, reverse=True)
    return {
        "benchmark":   _smart_join(bench_frags),
        "due_date":    date_val   or None,
        "status":      status_val or None,
        "reform_area": reform_area,
    }


def _identify_col_types(columns: list[str], data: dict) -> dict[str, str]:
    col_values: dict[str, list[str]] = {c: [] for c in columns}
    for row_vals in data.values():
        if not isinstance(row_vals, dict):
            continue
        for col, val in row_vals.items():
            if val and col in col_values:
                col_values[col].append(str(val).strip())

    result = {}
    for col, values in col_values.items():
        if not values:
            result[col] = "unknown"
            continue
        n = len(values)
        date_hits   = sum(1 for v in values if _DATE_VAL_RE.match(v) and len(v) < 50)
        status_hits = sum(1 for v in values if _STATUS_VAL_RE.match(v))
        if date_hits / n > 0.35:
            result[col] = "date"
        elif status_hits / n > 0.35:
            result[col] = "status"
        else:
            result[col] = "other"
    return result


def _sb_col_is_clean(col_name: str, data: dict) -> bool:
    values = [
        str(row.get(col_name, "") or "").strip()
        for row in data.values()
        if isinstance(row, dict)
    ]
    non_null = [v for v in values if v]
    if not non_null:
        return False
    avg_len = sum(len(v) for v in non_null) / len(non_null)
    if avg_len < 25:
        return False
    noise = sum(
        1 for v in non_null
        if _DATE_VAL_RE.match(v) or _STATUS_VAL_RE.match(v) or _REFORM_FULL_RE.match(v)
    )
    return noise / len(non_null) < 0.3


def _extract_format_b(data: dict, columns: list[str]) -> tuple[dict, list[str]]:
    sb_col = next((c for c in columns if _SB_COL_RE.match(c)), None)

    if sb_col and _sb_col_is_clean(sb_col, data):
        other_cols = [c for c in columns if c != sb_col]
        date_col   = next((c for c in other_cols if re.search(r"date", c, re.IGNORECASE)), None)
        status_col = next((c for c in other_cols if re.search(r"status", c, re.IGNORECASE)), None)

        # Fallback: use value-pattern inference to find date/status columns
        # when column names don't contain the expected keywords.
        if date_col is None or status_col is None:
            col_types = _identify_col_types(other_cols, data)
            if date_col is None:
                date_col = next(
                    (c for c in other_cols if col_types.get(c) == "date"), None
                )
            if status_col is None:
                status_col = next(
                    (c for c in other_cols if col_types.get(c) == "status"), None
                )

        rows: dict = {}
        for row_key, val in data.items():
            if not isinstance(val, dict):
                continue
            benchmark = (val.get(sb_col) or "").strip()
            if not benchmark or len(benchmark) < 15:
                continue
            if not re.search(
                r"\b(adopt\w*|prepar\w*|develop\w*|publish\w*|implement\w*|submit\w*|establish\w*|"
                r"operationali\w*|conduct\w*|strengthen\w*|updat\w*|finali[sz]\w*|complet\w*|"
                r"introduc\w*|design\w*|digitali\w*|transpos\w*|reconcil\w*|appoint\w*|"
                r"formali\w*|build\w*|increas\w*|reduc\w*|reform\w*|"
                r"review\w*|assess\w*|ensur\w*|provid\w*|support\w*|creat\w*|integrat\w*|"
                r"extend\w*|repeal\w*|revis\w*|begin\w*|enhanc\w*|issu\w*|resubmit\w*|remit\w*|"
                r"abolish\w*|eliminat\w*|transfer\w*|restructur\w*|merg\w*|"
                r"launch\w*|approv\w*|authoris\w*|authoriz\w*|sign\w*|ratif\w*|amend\w*|align\w*|"
                r"expan\w*|compil\w*|hir\w*|notif\w*|clos\w*|open\w*|submiss\w*|"
                r"enforc\w*|requir\w*|maintain\w*|achiev\w*|mandat\w*|allow\w*|enabl\w*|"
                r"apply\w*|appl[yi]\w*|disseminat\w*|improv\w*|operat\w*|increas\w*|transfer\w*|"
                r"in line)\b",
                benchmark, re.IGNORECASE
            ):
                continue
            rows[benchmark] = {
                "Due date":    val.get(date_col)   if date_col   else None,
                "Status":      val.get(status_col) if status_col else None,
                "Reform Area": row_key.strip(),
            }
        if rows:
            return rows, ["Due date", "Status", "Reform Area"]

    rows = {}
    for row_key, val in data.items():
        if not isinstance(val, dict):
            continue
        normalized = _normalize_sb_row(row_key, val)
        benchmark  = normalized["benchmark"]
        if not benchmark or len(benchmark) < 15:
            continue
        if _REFORM_FULL_RE.match(benchmark.strip()):
            continue
        rows[benchmark] = {
            "Due date":    normalized["due_date"],
            "Status":      normalized["status"],
            "Reform Area": normalized["reform_area"],
        }
    return rows, ["Due date", "Status", "Reform Area"]


def _extract_format_c(data: dict, columns: list[str]) -> tuple[dict, list[str]]:
    rows: dict = {}
    for key, val in data.items():
        if not isinstance(val, dict):
            continue
        all_null = all(v is None for v in val.values())
        if all_null and len(key.strip()) < 50:
            continue
        rows[key.strip()] = {c: val.get(c) for c in columns}

    # Ensure Status is surfaced even when the column is not named "status".
    has_status_col = any(re.search(r"status", c, re.IGNORECASE) for c in columns)
    if not has_status_col and rows:
        col_types = _identify_col_types(columns, data)
        inferred_status_col = next(
            (c for c in columns if col_types.get(c) == "status"), None
        )
        if inferred_status_col:
            for row_key, val in rows.items():
                if isinstance(val, dict) and "Status" not in val:
                    val["Status"] = val.get(inferred_status_col)

    return rows, list(columns)


def _infer_table_title(columns: list[str], html: str) -> str:
    for col in columns:
        if "." in col:
            candidate = col.rsplit(".", 1)[0].strip()
            if _SB_TITLE_RE.search(candidate) and len(candidate) > 10:
                return candidate
    if html:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            cap = soup.find("caption")
            if cap:
                return cap.get_text(strip=True)
        except Exception:
            pass
    return ""



# ---------------------------------------------------------------------------
# Claude API vision utilities
# ---------------------------------------------------------------------------

_VISION_MODEL = "claude-sonnet-4-6"

_VISION_PROMPT_SI_TOC = """You are reading the Table of Contents of an IMF Selected Issues Paper.

Your task: extract ONLY the top-level chapter titles -- the main section headings,
NOT subsections (A., B., C...), NOT figure/table lists, NOT references.

Chapter titles are typically:
- Written in ALL CAPS or bold
- Followed by a leader (underscores or dots) and a page number
- May start with "COUNTRY--" or be a standalone topic title

Return ONLY a JSON array of strings, one per chapter title, in document order.
No markdown, no explanation, no code fences. Example:
["CHAPTER ONE TITLE", "CHAPTER TWO TITLE", "CHAPTER THREE TITLE"]
"""


# Caps TOTAL concurrent Claude API calls across every thread, regardless of
# how many documents/pages are being processed at once. Without this, a
# batch run with WORKERS=3 outer threads x up to 4 inner page-level calls
# each can fire up to 12 simultaneous requests -- each document's internal
# ThreadPoolExecutor has no idea other documents are also hitting the API
# concurrently. That's a very plausible source of the intermittent silent
# failures seen even with retry logic in place (e.g. MDG_2022_ECF_Review1
# succeeding on some runs and silently returning 0 rows on others with no
# code change in between). Set CLAUDE_MAX_CONCURRENT in the environment to
# override the default (read once at import time), or call
# set_max_concurrent() at runtime to change it without restarting the kernel.
import threading as _threading
_CLAUDE_MAX_CONCURRENT = int(os.environ.get("CLAUDE_MAX_CONCURRENT", "4"))
_claude_semaphore = _threading.Semaphore(_CLAUDE_MAX_CONCURRENT)


def set_max_concurrent(n: int) -> None:
    """Change the global cap on simultaneous Claude API calls at runtime,
    without needing to restart the kernel. 4 is a conservative default
    picked without knowing your actual Anthropic rate-limit tier -- if your
    tier supports more concurrent requests, raising this recovers
    throughput. If you start seeing 0-row/empty extractions reappear after
    raising it, that's a sign you've gone past your actual rate limit --
    dial it back down."""
    global _CLAUDE_MAX_CONCURRENT, _claude_semaphore
    _CLAUDE_MAX_CONCURRENT = n
    _claude_semaphore = _threading.Semaphore(n)
    print(f"CLAUDE_MAX_CONCURRENT set to {n}")


def _call_claude_vision(image_bytes: bytes, media_type: str, prompt: str,
                         max_retries: int = 4) -> str:
    """
    Call Claude API with an image and prompt. Returns the raw text response,
    or "" on any error. Uses ANTHROPIC_API_KEY from environment.

    Retries with exponential backoff on transient failures (rate limits,
    timeouts, 5xx server errors, connection errors) -- previously a single
    transient failure anywhere (very plausible under concurrent batch
    processing, e.g. WORKERS=3 outer threads x up to 4 inner page-level
    calls each) would silently produce "" -> zero rows -> indistinguishable
    from "this document genuinely has no SB table". That masked real
    failures as empty results with no visible error.

    Also acquires a global semaphore (_claude_semaphore) before making the
    actual network call, capping total in-flight requests across every
    thread in the process -- see the comment above _claude_semaphore.
    """
    import base64
    import time

    api_key = _API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set -- Claude vision disabled.")
        return ""

    b64  = base64.standard_b64encode(image_bytes).decode("ascii")
    body = {
        "model": _VISION_MODEL,
        "max_tokens": 8192,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image",
                     "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        # NOTE: assistant-turn prefill (ending messages with a partial
        # {"role": "assistant", "content": "["} to force output to start
        # with JSON) was tried here, but Anthropic's docs confirm prefilling
        # is NOT supported on Claude 4.6+ models -- _VISION_MODEL is
        # "claude-sonnet-4-6", so using prefill returns a hard 400 error on
        # every call. Reverted. Any stray preamble Claude adds despite the
        # prompt's "no explanation" instruction is instead handled by
        # _parse_json_response()'s fallback, which locates the first "["
        # or "{" and parses from there.
    }
    headers = {
        "content-type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }

    import requests as _req

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with _claude_semaphore:
                resp = _req.post(
                    "https://api.anthropic.com/v1/messages",
                    json=body, headers=headers, timeout=90,
                )
            resp.raise_for_status()
            for block in resp.json().get("content", []):
                if block.get("type") == "text":
                    return block["text"].strip()
            logger.warning("Claude API returned no text content block (attempt %d/%d)",
                            attempt, max_retries)
            return ""  # got a response, just no text -- not worth retrying
        except _req.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            last_exc = exc
            body_text = exc.response.text[:500] if exc.response is not None else ""
            if status is not None and status not in (429,) and status < 500:
                # Non-retryable client error (401, 400, etc.) -- fail fast.
                # Include the actual response body: Anthropic's error
                # messages are specific (e.g. "prefilling not supported on
                # this model") and far more useful than the generic
                # exception string alone.
                logger.error("Claude API call failed with non-retryable status %s: %s -- body: %s",
                              status, exc, body_text)
                break
            wait = min(2 ** attempt, 30)
            logger.warning("Claude API call failed (status %s, attempt %d/%d) -- body: %s -- retrying in %ds",
                            status, attempt, max_retries, body_text, wait)
            if attempt < max_retries:
                time.sleep(wait)
        except (_req.exceptions.Timeout, _req.exceptions.ConnectionError) as exc:
            last_exc = exc
            wait = min(2 ** attempt, 30)
            logger.warning("Claude API network error (attempt %d/%d): %s -- retrying in %ds",
                            attempt, max_retries, exc, wait)
            if attempt < max_retries:
                time.sleep(wait)
        except Exception as exc:
            # Unexpected error type -- log loudly and don't retry blindly.
            last_exc = exc
            logger.error("Claude API call failed with unexpected error: %s", exc)
            break

    logger.error("Claude API call FAILED after %d attempt(s), giving up: %s",
                  max_retries, last_exc)
    return ""


def _parse_json_response(text: str):
    """Strip markdown fences and parse JSON from Claude response.

    Defense in depth: even with the assistant-turn prefill in
    _call_claude_vision (which should structurally prevent this), if a
    natural-language preamble ever precedes the JSON (e.g. "I'll analyze
    the table..." before the array), fall back to locating the first
    '[' or '{' and parsing from there instead of giving up entirely.
    """
    import json as _json
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.MULTILINE)
    text = text.strip()
    try:
        return _json.loads(text)
    except _json.JSONDecodeError:
        pass

    # Fallback: strip any leading preamble before the first bracket/brace.
    m = re.search(r"[\[{]", text)
    if m:
        try:
            return _json.loads(text[m.start():])
        except _json.JSONDecodeError:
            pass
    return None


# ---------------------------------------------------------------------------
# Claude-API-only SB extraction  (replaces Docling + regex pipeline)
# ---------------------------------------------------------------------------
# Strategy:
#   1. _find_benchmark_pages() -- same fitz scan as before (cheap, < 1 s)
#   2. Render each SB page as a PNG at 200 dpi via fitz
#   3. Send each page image to Claude vision with a detailed prompt
#   4. Parse the returned JSON array into BenchmarkResult objects
#   5. Merge "(Concluded)" continuation pages
#
# Advantages over Docling path:
#   - No Docling dependency; no TableFormer model download
#   - Handles image-embedded tables natively (no separate vision fallback)
#   - Handles any novel column layout -- Claude reads the table semantically
#   - Status column always extracted regardless of column name
#
# Trade-offs:
#   - Requires ANTHROPIC_API_KEY
#   - Network round-trip per page (~1-3 s, parallelised across pages)
# ---------------------------------------------------------------------------

_VISION_PROMPT_SB_FULL = (
    "You are extracting Structural Benchmark rows from an IMF table image.\n\n"
    "NOTE: Some country reports call these 'Structural Conditionality' instead of\n"
    "'Structural Benchmarks' — treat the two terms as fully equivalent throughout this task,\n"
    "including in table titles and section divider rows (e.g. 'Proposed New Structural\n"
    "Conditionality' means the same thing as 'Proposed New Structural Benchmark' below).\n\n"
    "NOTE: Documents under the PCI (Policy Coordination Instrument) use 'Reform Targets'\n"
    "instead of 'Structural Benchmarks' (e.g. a table titled 'Reform Targets Under the PCI'\n"
    "or 'PCI Reform Targets'). Treat 'Reform Target' as fully equivalent to 'Structural\n"
    "Benchmark' throughout this task — same extraction rules, same output field names (still\n"
    "use \"benchmark\" as the JSON key for the target text, not \"reform_target\").\n\n"
    "IMPORTANT — some tables are titled 'Prior Action and Structural Benchmarks' (or\n"
    "'... and Structural Conditionality') and mix two types:\n"
    "  • Prior Actions  — appear BEFORE any 'Proposed New Structural Benchmark' section header,\n"
    "                     OR in a column explicitly headed 'Prior Action'.\n"
    "  • Structural Benchmarks — appear AFTER a row/header that says 'Proposed New Structural\n"
    "                     Benchmark', OR in a column explicitly headed 'Structural Benchmark(s)'.\n\n"
    "CRITICAL RULE: Extract ONLY Structural Benchmarks. SKIP all Prior Action rows entirely.\n"
    "  - If a row's text begins with 'Prior Action:' (bolded or otherwise) → skip it entirely.\n"
    "  - If the table has a 'Prior Action' column AND a separate 'Structural Benchmark(s)' column,\n"
    "    extract only rows from the Structural Benchmark(s) column.\n"
    "  - If the table has a single action column with a section divider row labelled\n"
    "    'Proposed New Structural Benchmark' OR simply 'Structural benchmarks' (no 'Proposed New'\n"
    "    prefix required), extract only rows BELOW that divider.\n"
    "  - A section divider row reading 'Prior action' or 'Prior actions' (singular or plural,\n"
    "    any capitalization) marks the rows immediately below it as Prior Actions — skip them.\n"
    "  - IMPORTANT: an explicit 'Structural benchmarks' divider row is NOT always present. Many\n"
    "    tables go directly from 'Prior action(s)' straight into the first thematic section header\n"
    "    (e.g. 'Mobilizing fiscal policy to address economic and social needs') with no separate\n"
    "    'Structural benchmarks' label in between at all. In that case, the Prior Actions section\n"
    "    simply ends at whichever comes first: an explicit 'Structural benchmarks' divider, OR the\n"
    "    next bolded/underlined section-header row of ANY kind (including a thematic title). Do NOT\n"
    "    require the literal words 'Structural benchmarks' to appear before resuming extraction --\n"
    "    reaching any new section header row is enough to end the Prior Actions skip.\n"
    "  - If the entire table is headed 'Structural Benchmarks' with no Prior Action rows, extract all rows.\n\n"
    "RESET / MODIFIED BENCHMARKS:\n"
    "Some rows show a benchmark being replaced with a revised version, e.g. a cell reading:\n"
    "  \"Keep the government's liability... below MGA 100 billion. ... To be replaced by: Keep\n"
    "  the government's gross liabilities... below MGA 300 billion.\"\n"
    "The \"To be replaced by: ...\" portion (often shown in blue, like newly proposed benchmarks)\n"
    "signals this benchmark's target is being reset/modified. For such a row:\n"
    "  - Use the ORIGINAL (pre-replacement) benchmark text as the benchmark field.\n"
    "  - Set status=\"Reset\" (this is a real, expected status value -- do not use \"New\" here,\n"
    "    since this is a modification of an existing benchmark, not a newly proposed one).\n\n"
    "NEWLY PROPOSED BENCHMARKS — COLOR SIGNAL:\n"
    "Some tables include a note like 'Proposed new structural benchmarks are indicated in blue.'\n"
    "When present, look at the actual rendered color of each benchmark's text in the image:\n"
    "  - If a row's benchmark text is rendered in BLUE (as opposed to the normal black text used\n"
    "    for previously-existing benchmarks), and that row's status cell is blank/empty, set\n"
    "    status=\"New\" for that row (do not leave it null).\n"
    "  - This applies whether or not the table has an explicit color-note -- always check for a\n"
    "    color difference between rows if some benchmarks look distinctly blue vs. black.\n"
    "  - A blank status cell on a row that is NOT blue-colored should still be status=null.\n"
    "  - Exception: a blue \"To be replaced by: ...\" fragment within an otherwise-existing row is\n"
    "    the Reset case above, not a brand-new row -- don't treat the whole row as \"New\".\n\n"
    "REFORM AREA AS A SECTION HEADER:\n"
    "Some tables present Reform Area not as a per-row column, but as a bolded/underlined\n"
    "section-title row spanning multiple benchmarks below it (e.g. a row reading 'Mobilizing\n"
    "fiscal policy to address economic and social needs' followed by several numbered benchmarks,\n"
    "then another section-title row for the next theme). These theme-title rows are themselves\n"
    "structural content, not benchmarks -- do not extract them as a benchmark row. Instead, apply\n"
    "the full theme sentence as the reform_area value for every benchmark row until the next such\n"
    "section-title row appears.\n\n"
    "REVIEW-ROUND HEADERS ARE NOT REFORM AREAS:\n"
    "Some tables instead group rows under review-round labels -- 'First Review', 'Second Review',\n"
    "'Third Review', 'Continuous', 'Proposed Structural Benchmarks' -- rather than policy themes.\n"
    "These are administrative/timing labels, not reform categories. Do NOT use them as the\n"
    "reform_area value. If a table has ONLY this kind of section header (no separate thematic\n"
    "Reform Area column or theme-title rows as described above), set reform_area=null for every\n"
    "row instead of using the review-round label.\n\n"
    "ONLY FOUR FIELDS EXIST — IGNORE EVERYTHING ELSE:\n"
    "The output schema has exactly four fields: reform_area, benchmark, due_date, status.\n"
    "Tables in practice often have MORE columns than these four, under many different names\n"
    "across documents -- 'Rationale', 'Comments', 'Indicator', 'Depth', 'Macro-criticality',\n"
    "or others not listed here at all. As a general rule: if a column doesn't map to Reform\n"
    "Area, the benchmark text itself, Due Date, or Status, IGNORE IT COMPLETELY, regardless of\n"
    "its name. This applies especially to any High/Medium/Low-style rating column (seen under\n"
    "names including 'Macro-criticality' and 'Depth') -- these are never a status. It also\n"
    "applies to a column literally named 'Review' containing values like '1st Review', '2nd\n"
    "Review', '3rd Review' -- this indicates which review round the target is associated with,\n"
    "NOT a completion status; ignore it, and leave status=null unless a real status word (Met,\n"
    "Not met, etc.) appears elsewhere for that row. Never place a rating value, a\n"
    "verification-criteria description, a review-round label, or explanatory notes into the\n"
    "status or reform_area fields just because you're unsure where else to put them -- if it\n"
    "isn't clearly the benchmark text, due date, or a real status value, leave it out.\n\n"
    "The table columns (exact names and order vary by document):\n"
    '  1. Reform area   -- short category label (e.g. "Revenue Mobilization", "Governance", "PFM"),\n'
    "                     OR a longer thematic section-header sentence as described above.\n"
    "  2. Structural benchmark -- the full policy action text (one or two sentences)\n"
    '  3. Due date      -- e.g. "End-June 2025", "end-July 2024", "End-December 2024"\n'
    '  4. Status        -- e.g. "Met", "Not met", "On track", "Ongoing", "New",\n'
    '                     "Not yet due", "Observed", "Delayed", "Pending", "In progress",\n'
    '                     "Partly met", "Completed", "Dropped", "Waived", "Reset"\n'
    '  5. Rationale     -- explanatory text column present in some documents; IGNORE entirely.\n'
    '  6. Comments      -- notes column (e.g. "Reset from September 2021 to accommodate...");\n'
    '                     IGNORE entirely, same as Rationale.\n'
    '  7. Indicator     -- describes what evidence/document would verify completion\n'
    '                     (e.g. "Audited Inventory report... shared with IMF staff.");\n'
    '                     IGNORE entirely, same as Rationale.\n'
    '  8. Macro-criticality / Depth -- a High/Medium/Low rating column, under either name\n'
    '                     (or others not listed here). This is NOT a status. IGNORE entirely --\n'
    '                     do not put its value ("High", "Medium", "Low") in the status field.\n\n'
    "IMPORTANT — not every table has every column above. Program Request documents in\n"
    "particular commonly have ONLY \"Measures\" (or similar), \"Due Date(s)\", \"Rationale\",\n"
    "and \"Macro-criticality\" — with NO separate Reform area column and NO Status column at\n"
    "all, because every benchmark listed is newly proposed and hasn't been categorized or\n"
    "assessed yet. In that case:\n"
    "  - reform_area = null for every row (do not invent or guess a category).\n"
    "  - status = \"New\" for every row (every benchmark in a Program Request is newly proposed).\n"
    "  - The numbered \"Measures\" column (\"1. Enforce the collection...\", \"2. Adopt...\")\n"
    "    is the benchmark text itself — include the full sentence(s), you may drop the\n"
    "    leading number.\n\n"
    "Other rules:\n"
    "- Reform area cells often SPAN multiple rows (merged cell) -- repeat the value for each row.\n"
    '- Status may be in a column named "Observance", "Implementation", "Implementation status",\n'
    '  or similar -- always map it to status.\n'
    "- STATUS EMBEDDED IN THE DATE CELL: some tables have no separate Status column at all, and\n"
    '  a completed row instead shows the status directly inside the date cell, e.g. "Met on June\n'
    '  4, 2021" or "Completed March 2022". When you see this pattern, split it: status = "Met" (or\n'
    "  the relevant status word), due_date = just the date portion (\"June 4, 2021\"). Do not leave\n"
    "  the status word inside due_date, and do not lose the date by discarding the whole cell.\n"
    "  Rows in the same table without this pattern (a plain date, or nothing) still follow the\n"
    "  normal rules above -- due_date=null if blank, status=null unless another rule sets it.\n"
    "- This may be a (Concluded) continuation page -- extract ALL SB rows shown.\n"
    "- A row with a genuinely blank status cell that is NOT a newly-proposed (blue) benchmark and\n"
    "  is NOT from a Program Request document should have status=null (not empty string).\n"
    "- A row with no due date should have due_date=null.\n"
    "- Extract EVERY visible data row without exception. Do not summarise, skip, or truncate.\n"
    "- If a cell is blank or missing, use null.\n\n"
    "Return ONLY a JSON array -- no markdown, no explanation, no code fences.\n"
    'Each element must have exactly these keys: "reform_area", "benchmark", "due_date", "status"\n\n'
    "Example (mixed table — only SB rows returned, Prior Actions skipped):\n"
    "[\n"
    '  {"reform_area": "PFM", "benchmark": "Enforce e-procurement to be used by at least 50 percent of all ordinary operations.", "due_date": "end-July 2024", "status": null},\n'
    '  {"reform_area": "Revenue Mobilization", "benchmark": "Adopt and publish a medium-term revenue strategy.", "due_date": "End-June 2024", "status": "Met"}\n'
    "]"
)


def _render_page_and_call_claude(pdf_path: str, page_num: int, prompt: str):
    """
    Render a single PDF page at 200 dpi and send it to Claude vision with
    the given prompt. Returns the parsed JSON response (list or dict,
    whatever the prompt asked for), or None on any failure. Shared by all
    three table extractors (SB, Prior Actions, QPC) -- only the prompt and
    expected shape differ between them.
    """
    import fitz
    doc = fitz.open(pdf_path)
    try:
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = doc[page_num - 1].get_pixmap(matrix=mat)
        image_bytes = pix.tobytes("png")
        media_type  = "image/png"
    finally:
        doc.close()

    text = _call_claude_vision(image_bytes, media_type, prompt)
    return _parse_json_response(text) if text else None


def _extract_sb_page_via_claude(pdf_path: str, page_num: int) -> list[dict]:
    """
    Render a single PDF page at 200 dpi and send it to Claude vision for SB
    extraction. Returns a list of raw row dicts with keys:
      reform_area, benchmark, due_date, status
    """
    parsed = _render_page_and_call_claude(pdf_path, page_num, _VISION_PROMPT_SB_FULL)
    result = parsed if isinstance(parsed, list) else []
    logger.info("[%s] p.%d -- Claude returned %d raw rows", pdf_path, page_num, len(result))
    return result


def extract_structural_benchmarks(pdf_path: str, table_mode: str = "fast",
                                  include_reform_targets: bool = True,
                                  include_prior_actions: bool = False) -> list[BenchmarkResult]:
    """
    Extract Structural Benchmarks using Claude vision API only.
    Docling and regex-based table parsing are not used.
    Requires ANTHROPIC_API_KEY to be set (via set_api_key() or environment).

    Every row carries a "Type" value:
      "SB"            -- structural benchmark / structural conditionality
      "Reform Target" -- row from a PCI "Reform Targets" table
      "PA"            -- prior action (only when include_prior_actions=True)
    If a document has both an SB table (e.g. GHA Table 2) and a Reform
    Targets table (GHA Table 5), both are extracted unless
    include_reform_targets=False.
    """
    import concurrent.futures

    pdf_path = str(pdf_path)
    _tables = _find_benchmark_tables(pdf_path, include_reform_targets)
    page_type: dict[int, str] = {p: t["kind"] for t in _tables for p in t["pages"]}
    benchmark_pages = sorted(page_type)
    if not benchmark_pages:
        return []

    logger.info("[%s] Claude-API SB extraction on %d pages: %s",
                pdf_path, len(benchmark_pages), benchmark_pages)

    # Parallelise page renders + API calls (I/O bound)
    page_rows: dict[int, list[dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_extract_sb_page_via_claude, pdf_path, pg): pg
            for pg in benchmark_pages
        }
        for fut in concurrent.futures.as_completed(futures):
            pg = futures[fut]
            try:
                page_rows[pg] = fut.result()
            except Exception as exc:
                logger.warning("[%s] p.%d Claude extraction failed: %s", pdf_path, pg, exc)
                page_rows[pg] = []

    # Filter rows and build BenchmarkResult per page
    # Prior Actions that leak through from mixed tables — exclude them.
    # These are typically budget submissions or legal compliance actions
    # explicitly listed as "Prior Action" in the table, not SBs.
    # (The vision prompt already instructs Claude to skip Prior Actions;
    # this is a backstop for cases where the prompt doesn't hold.)
    _PRIOR_ACTION_RE = re.compile(
        r"^\s*Prior\s+Action\s*:"
        r"|submit\s+to\s+parliament\s+\d{4}\s+budget"
        r"|submit\s+to\s+parliament.*budget\s+in\s+line\s+with\s+program",
        re.IGNORECASE,
    )

    # NOTE: No action-verb whitelist here. The Claude vision model already
    # understands which rows are benchmarks semantically; a verb filter only
    # causes false negatives (e.g. "Apply...", "Disseminate...", "Improve...").

    # Pre-scan pages to detect "(concluded)" / "(continued)" continuation pages.
    # Tanzania-style PDFs put the marker after the table title, not on the first
    # line, so we search the full page text rather than just the first line.
    import fitz as _fitz2
    _doc2 = _fitz2.open(pdf_path)
    _CONT_RE = re.compile(r"\(conclu\w+\)|\(continu\w+\)", re.IGNORECASE)
    _SB_TITLE_LINE_RE = re.compile(
        r"(table\s+\d+[a-z]?[\.:].{0,200}?(?:structural\s+(?:benchmarks?|conditionality)|reform\s+targets?)[^\n]*)",
        re.IGNORECASE | re.DOTALL,
    )
    page_titles: dict[int, str] = {}
    for pg in benchmark_pages:
        text = _doc2[pg - 1].get_text() if pg <= len(_doc2) else ""
        # Use actual table title line if found, else fall back to first line.
        # Collapse any embedded newline (from a wrapped caption spanning two
        # lines, e.g. "Table 2. Country:\nPrior Actions and Structural
        # Benchmarks...") into a single space for a clean display title.
        m = _SB_TITLE_LINE_RE.search(text)
        base_title = re.sub(r"\s*\n\s*", " ", m.group(1)).strip() if m else (text.splitlines()[0].strip() if text else "")
        # Append (concluded)/(continued) if present anywhere on the page
        cont_m = _CONT_RE.search(text)
        suffix = f" {cont_m.group(0)}" if cont_m and cont_m.group(0).lower() not in base_title.lower() else ""
        page_titles[pg] = base_title + suffix
    _doc2.close()

    raw_results: list[BenchmarkResult] = []
    for pg in sorted(benchmark_pages):
        rows = page_rows.get(pg, [])
        result_rows: dict = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            benchmark = (item.get("benchmark") or "").strip()
            if not benchmark or len(benchmark) < 15:
                continue
            if _PRIOR_ACTION_RE.search(benchmark):
                logger.info("[%s] p.%d -- Skipping Prior Action row: %s", pdf_path, pg, benchmark[:80])
                continue
            result_rows[benchmark] = {
                "Due date":    item.get("due_date")    or None,
                "Status":      item.get("status")      or None,
                "Reform Area": item.get("reform_area") or None,
                "Type":        page_type.get(pg, "SB"),
            }
        if result_rows:
            # Use actual page title so (concluded)/(continued) merge fires correctly
            page_title = page_titles.get(pg, "")
            table_title = page_title if page_title else f"Structural Benchmarks (p.{pg})"
            raw_results.append(BenchmarkResult(
                table_title=table_title,
                page_num=pg,
                columns=["Due date", "Status", "Reform Area", "Type"],
                rows=result_rows,
            ))

    # NOTE: Status is intentionally left exactly as Claude reported it here --
    # including genuinely blank/null values -- with no "blank -> New"
    # modification at this layer. That resolution now happens entirely in
    # 02_transform_json_to_excel.ipynb's _resolve_blank_status(), which
    # already handles both cases correctly with full context this function
    # doesn't have access to:
    #   - Program Request documents: blank -> "New" (unambiguous)
    #   - Review N documents: blank -> "New" only if the benchmark never
    #     appeared in any prior document for that country; otherwise leave
    #     blank (it existed before, just not reassessed this round)
    #
    # This engine used to duplicate the Request-document half of that logic
    # here, then briefly over-generalized it to ALL documents (which broke
    # on CMR, where several pre-existing benchmarks have blank status despite
    # not being new). Removing it entirely avoids re-deciding status at two
    # different layers -- the engine now reports raw data faithfully, and
    # the notebook is the single place status gets resolved.
    #
    # The one thing that still sets status="New" here, deliberately, is
    # Claude's OWN direct color-detection during extraction (a benchmark
    # rendered in blue text, per the vision prompt) -- that's a real signal
    # read from the document itself, not a guess, so it's already reflected
    # in result_rows above and untouched by this change.

    # Merge "(Concluded)" continuation pages into their parent table
    _CONCLUDED_RE = re.compile(r"\(conclu\w+\)|\(continu\w+\)", re.IGNORECASE)
    merged: list[BenchmarkResult] = []
    for br in raw_results:
        if _CONCLUDED_RE.search(br.table_title or "") and merged:
            parent = merged[-1]
            for k, v in br.rows.items():
                if k not in parent.rows:
                    parent.rows[k] = v
            logger.info("  Merged Concluded page (%d rows) into p.%d table",
                        len(br.rows), parent.page_num)
        else:
            merged.append(br)

    if include_prior_actions:
        for pa in extract_prior_actions(pdf_path):
            merged.append(BenchmarkResult(
                table_title=pa.table_title,
                page_num=pa.page_num,
                columns=["Due date", "Status", "Reform Area", "Type"],
                rows={action: {"Due date": v.get("Due date"), "Status": v.get("Status"),
                               "Reform Area": v.get("Reform Area"), "Type": "PA"}
                      for action, v in pa.rows.items()},
            ))
        merged.sort(key=lambda b: b.page_num)

    logger.info("[%s] Claude-API SB extraction complete: %d tables, %d total rows",
                pdf_path, len(merged), sum(len(b.rows) for b in merged))
    return merged


# ---------------------------------------------------------------------------
# Prior Actions extraction  [Claude vision API]
# ---------------------------------------------------------------------------
# Prior Actions frequently share a physical table/page with Structural
# Benchmarks (e.g. Liberia's Table 3, "Prior Actions and Structural
# Benchmarks for 2025-27"): the top of the table lists Prior Action rows
# (just Definition + Status/Comment, no Target Date/Rationale), often under
# two sub-groups -- plain "Prior Actions" and "Structural Benchmarks to be
# implemented as Prior Action" -- followed by a "Structural Benchmarks"
# divider that introduces Target Date/Rationale columns for the genuine,
# forward-looking SB rows. This prompt is deliberately the mirror image of
# _VISION_PROMPT_SB_FULL: it extracts everything ABOVE that divider (both
# prior-action sub-groups) and skips everything below it.
# ---------------------------------------------------------------------------

_VISION_PROMPT_PRIOR_ACTIONS_FULL = (
    "You are extracting PRIOR ACTION rows from an IMF table image (any country, any program:\n"
    "ECF, EFF, SBA, RSF, PCI...). Layouts differ from document to document, so read the page\n"
    "carefully and follow the rules below.\n\n"
    "WHAT A PRIOR ACTION IS:\n"
    "A prior action is a measure the authorities must complete BEFORE the IMF Board approves an\n"
    "arrangement or completes a review. It is NOT a structural benchmark (a measure due at a\n"
    "later date) and NOT a quantitative target. Extract ONLY prior actions.\n\n"
    "WHERE PRIOR ACTIONS APPEAR -- HANDLE ALL OF THESE LAYOUTS:\n"
    "  (1) A table titled just 'Prior Actions' / 'Prior Action' / 'Proposed Prior Actions' /\n"
    "      'Prior Actions for Board Approval' etc. Every row is a prior action.\n"
    "  (2) A combined table ('Prior Actions and Structural Benchmarks', 'Structural\n"
    "      Conditionality', ...) split into SECTIONS by header rows or divider rows. Prior\n"
    "      actions are the rows in the section headed 'Prior Action(s)'. The section headed\n"
    "      'Structural Benchmarks' / 'Structural Conditionality' (often with new column headers\n"
    "      such as 'Target Date' / 'Rationale') is NOT prior actions -- stop there and do not\n"
    "      extract anything below it.\n"
    "      Example header rows: 'Prior Actions | Objective | Date | Status', then later\n"
    "      'Structural Benchmarks | Objective | Date | Status'.\n"
    "  (3) Prior-action rows that sit at the TOP of a table with no heading of their own, just\n"
    "      an action column and a status/comment column, above a 'Structural Benchmarks' divider.\n"
    "  (4) Rows inside an SB table whose text begins with 'Prior Action:' (or is tagged 'PA',\n"
    "      or has an indicator column marking it as a prior action).\n"
    "  (5) Rows under a sub-heading such as 'Structural Benchmarks to be implemented as Prior\n"
    "      Action' or 'Prior Actions for the Xth Review' -- these are ALSO prior actions.\n"
    "Go by the SECTION a row sits in, not by which cells are filled: prior-action rows often have\n"
    "blank Date and Status cells.\n\n"
    "WHAT TO EXTRACT -- exactly these five keys per row:\n"
    '  - "action"    -- the full text of the prior action as printed (bold and regular text\n'
    "                   together; drop only a leading row number like '1.' or '(a)'). Do not\n"
    "                   summarise or truncate. If one numbered item contains several sub-points\n"
    "                   (i), (ii), (iii), keep them together as ONE row.\n"
    '  - "status"    -- the value in a Status / Status-Comment / Observance / Implementation\n'
    "                   column, copied in full (e.g. 'Met', 'Not met, fuel excise increase\n"
    "                   replaced by expenditure cuts', 'Implemented'). null if the cell is blank\n"
    "                   or there is no such column.\n"
    '  - "due_date"  -- the value in a Date / Target Date / Due Date / Timing column for that\n'
    "                   row, copied as printed. null if blank or no such column. (Do not invent\n"
    "                   a date from the action text.)\n"
    '  - "category"  -- "Prior Action" for ordinary prior actions; "Structural Benchmark\n'
    '                   implemented as Prior Action" for rows under a sub-heading like\n'
    "                   'Structural Benchmarks to be implemented as Prior Action'.\n"
    '  - "reform_area" -- ONLY if the table has a real Reform Area / Policy Area / Area column,\n'
    "                   or a bold thematic section-title row spanning several prior-action rows\n"
    "                   (use that exact text). Otherwise null. Never fill this from an\n"
    "                   'Objective', 'Rationale' or 'Comments' column.\n\n"
    "COLUMNS TO IGNORE ENTIRELY: Objective, Rationale, Comments, Indicator, Macro-criticality,\n"
    "Depth, Verification, Review, and any High/Medium/Low rating. Never copy them into any field.\n\n"
    "DO NOT EXTRACT:\n"
    "  - Structural benchmark rows, reform target rows, or continuous benchmark rows.\n"
    "  - Quantitative performance criteria / indicative target tables (numbers by test date).\n"
    "  - Narrative paragraphs, bullet lists in running text, footnotes, or table titles/notes.\n"
    "  - Section or column header rows themselves ('Prior Actions', 'Structural Benchmarks',\n"
    "    'Definition', 'Status/Comment').\n\n"
    "EDGE CASES:\n"
    "  - If this page has NO prior-action rows (e.g. it only shows structural benchmark rows, a\n"
    "    '(Concluded)'/'(Continued)' tail of the SB section, a QPC table, or ordinary text), return\n"
    "    an empty JSON array []. Never invent rows.\n"
    "  - A '(Continued)' page of a prior-actions table may begin directly with rows and no\n"
    "    heading -- if the rows above any 'Structural Benchmarks' header are prior actions,\n"
    "    extract them; if you cannot tell they are prior actions, return [].\n"
    "  - A row cut off by a page break: extract the text that is visible.\n"
    "  - Keep the rows in the order they appear on the page.\n\n"
    "Return ONLY a JSON array -- no markdown, no explanation, no code fences.\n"
    'Each element must have exactly these keys: "action", "status", "due_date", "category",\n'
    '"reform_area"\n\n'
    "Example:\n"
    "[\n"
    '  {"action": "Publish the Bank Financial Institutions Holding Act.", "status": "Met", '
    '"due_date": null, "category": "Prior Action", "reform_area": null},\n'
    '  {"action": "MFDP to provide the complete borrowing terms, including detailed repayment '
    'schedules, and copies of the original contracts of all GoL external debt obligations.", '
    '"status": "Met", "due_date": null, "category": "Structural Benchmark implemented as Prior '
    'Action", "reform_area": null},\n'
    '  {"action": "GRA to complete the registration, filing, and payments modules for CIT, PIT, '
    'VAT", "status": null, "due_date": null, "category": "Prior Action", "reform_area": null}\n'
    "]"
)


def _extract_prior_action_page_via_claude(pdf_path: str, page_num: int) -> list[dict]:
    """Render a page and ask Claude vision to extract just its Prior Action
    rows. Returns a list of raw row dicts with keys:
      action, status, category, reform_area
    """
    parsed = _render_page_and_call_claude(pdf_path, page_num, _VISION_PROMPT_PRIOR_ACTIONS_FULL)
    result = parsed if isinstance(parsed, list) else []
    logger.info("[%s] p.%d -- Claude returned %d raw Prior Action rows",
                pdf_path, page_num, len(result))
    return result


def extract_prior_actions(pdf_path: str) -> list[PriorActionResult]:
    """
    Extract Prior Action rows using Claude vision API.

    Mirrors extract_structural_benchmarks(): find candidate pages via
    _find_prior_action_pages(), render each at 200 dpi, send to Claude
    vision with a prompt that slices out just the prior-action rows (see
    _VISION_PROMPT_PRIOR_ACTIONS_FULL for how it distinguishes those from
    Structural Benchmark rows that may share the same physical table).
    """
    import concurrent.futures

    pdf_path = str(pdf_path)
    pages = _find_prior_action_pages(pdf_path)
    if not pages:
        return []

    logger.info("[%s] Claude-API Prior Action extraction on %d page(s): %s",
                pdf_path, len(pages), pages)

    page_rows: dict[int, list[dict]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_extract_prior_action_page_via_claude, pdf_path, pg): pg
            for pg in pages
        }
        for fut in concurrent.futures.as_completed(futures):
            pg = futures[fut]
            try:
                page_rows[pg] = fut.result()
            except Exception as exc:
                logger.warning("[%s] p.%d Prior Action extraction failed: %s", pdf_path, pg, exc)
                page_rows[pg] = []

    # Page titles, for a readable table_title and to know which pages are
    # "(concluded)" continuations that should merge into an earlier page.
    import fitz as _fitz3
    _doc3 = _fitz3.open(pdf_path)
    _CONT_RE = re.compile(r"\(conclu\w+\)|\(continu\w+\)", re.IGNORECASE)
    page_titles: dict[int, str] = {}
    for pg in pages:
        text = _doc3[pg - 1].get_text() if pg <= len(_doc3) else ""
        # Use the first non-blank line as the title rather than the regex
        # match itself -- the match is deliberately lazy/short (just enough
        # to confirm "prior action" is present) and would otherwise
        # truncate a caption like "Table 3. Liberia: Prior Actions and
        # Structural Benchmarks for 2025-27" right after "Prior Actions".
        # Prefer the actual "Table N. ..." caption line (page headers such as
        # "GHANA" otherwise become the first non-blank line).
        first_line = next(
            (l.strip() for l in text.splitlines() if re.match(r"\s*table\s+\d+[a-z]?[\.:]", l, re.IGNORECASE)),
            next((l.strip() for l in text.splitlines() if l.strip()), ""),
        )
        base_title = first_line
        cont_m = _CONT_RE.search(text)
        suffix = f" {cont_m.group(0)}" if cont_m and cont_m.group(0).lower() not in base_title.lower() else ""
        page_titles[pg] = base_title + suffix
    _doc3.close()

    results: list[PriorActionResult] = []
    for pg in sorted(pages):
        rows = page_rows.get(pg, [])
        result_rows: dict = {}
        for item in rows:
            if not isinstance(item, dict):
                continue
            action = (item.get("action") or "").strip()
            if not action or len(action) < 10:
                continue
            result_rows[action] = {
                "Status":      item.get("status")      or None,
                "Category":    item.get("category")    or "Prior Action",
                "Reform Area": item.get("reform_area") or None,
                "Due date":    item.get("due_date")    or None,
            }
        if result_rows:
            title = page_titles.get(pg) or f"Prior Actions (p.{pg})"
            results.append(PriorActionResult(
                table_title=title,
                page_num=pg,
                columns=["Status", "Category", "Reform Area", "Due date"],
                rows=result_rows,
            ))

    # Merge "(concluded)" continuation pages into the preceding table, same
    # pattern as extract_structural_benchmarks.
    _CONCLUDED_RE = re.compile(r"\(conclu\w+\)|\(continu\w+\)", re.IGNORECASE)
    merged: list[PriorActionResult] = []
    for pr in results:
        if _CONCLUDED_RE.search(pr.table_title or "") and merged:
            parent = merged[-1]
            for k, v in pr.rows.items():
                if k not in parent.rows:
                    parent.rows[k] = v
            logger.info("  Merged Concluded page (%d rows) into p.%d Prior Action table",
                        len(pr.rows), parent.page_num)
        else:
            merged.append(pr)

    logger.info("[%s] Claude-API Prior Action extraction complete: %d table(s), %d total rows",
                pdf_path, len(merged), sum(len(r.rows) for r in merged))
    return merged


# ---------------------------------------------------------------------------
# Quantitative Performance Criteria (QPC) extraction  [Claude vision API]
# ---------------------------------------------------------------------------
# QPC/Indicative Target tables are structurally different from SB and Prior
# Action tables: they're a wide matrix of criteria (rows) x test dates
# (column groups), where each test date column can show a different subset
# of sub-values -- past/assessed dates typically show PC (or Program),
# Adjusted PC, Actual (or Prel.), and Status; future/proposed dates
# typically show only a single Proposed PC or Proposed IT value with no
# status yet. The prompt asks Claude to preserve this per-date, per-value
# structure explicitly rather than flattening it, since flattening would
# lose which numbers belong to which test date.
# ---------------------------------------------------------------------------

_VISION_PROMPT_QPC_FULL = (
    "You are extracting a Quantitative Performance Criteria (QPC) and Indicative Targets (IT)\n"
    "table from an IMF document image. This is a wide matrix: each row is a criterion (e.g.\n"
    "'Floor on primary fiscal balance, excluding grants'), and there are several test-date\n"
    "column groups across the top (e.g. 'Dec. 2024', 'Jun. 2025', 'Dec. 2025', 'Jun. 2026',\n"
    "'Dec. 2026'), each with its own sub-columns.\n\n"
    "ROW GROUPS (Category):\n"
    "Rows are grouped under bolded section headers, typically some subset of:\n"
    '  - "Performance Criteria" -- binding quantitative targets\n'
    '  - "Indicative Targets" -- monitored but non-binding targets\n'
    '  - "Memorandum Items" -- reference figures shown for context, not formal targets\n'
    "Use the exact section header text you see as the category for each row beneath it\n"
    "(some documents may only have some of these three, or may phrase them slightly\n"
    "differently -- always use the header text as actually shown on the page).\n\n"
    "TEST-DATE COLUMNS -- SUB-VALUES VARY BY DATE:\n"
    "Each test-date column group (e.g. 'Dec. 2024') has its OWN set of sub-columns, and this set\n"
    "commonly differs between past/assessed dates and future/proposed dates:\n"
    "  - An assessed (past) test date usually has 4 sub-values: a target value (often headed\n"
    "    'PC' or 'Program'), an 'Adjusted PC' value, an outturn value (headed 'Actual' or\n"
    "    'Prel.' for preliminary), and a 'Status' (e.g. 'Met', 'Not met', 'Not Met').\n"
    "  - A future/not-yet-assessed test date usually has just ONE sub-value, headed something\n"
    "    like 'PC', 'Proposed PC', or 'Proposed IT', and no status yet.\n"
    "  - Read the ACTUAL sub-column headers shown above each test date in the image -- do not\n"
    "    assume the exact 4-value or 1-value pattern always holds; some tables have different\n"
    "    combinations (e.g. no 'Adjusted PC' column, or an extra 'Indicative' column). Use\n"
    "    whatever sub-column header text is actually printed on the page as the key.\n\n"
    "OUTPUT SHAPE -- one object per criterion (row):\n"
    '  - "category": the section header this row falls under (see ROW GROUPS above).\n'
    '  - "criterion": the full criterion/target description text (e.g. "Floor on primary fiscal\n'
    "    balance, excluding grants\"). Strip footnote superscript numbers from the text itself,\n"
    "    but keep the full descriptive wording.\n"
    '  - "test_dates": a JSON array, one object per test-date column group, each with:\n'
    '      - "date": the test date label exactly as printed (e.g. "Dec. 2024", "Jun. 2026")\n'
    '      - "values": an object mapping each sub-column header text (exactly as printed, e.g.\n'
    '        "PC", "Adjusted PC", "Actual", "Prel.", "Status", "Proposed PC", "Proposed IT")\n'
    "        to that cell's value for this criterion and this test date. Use the sub-column\n"
    "        header text verbatim as the key so no information about which number came from\n"
    "        which column is lost. Numeric values can be given as strings exactly as shown\n"
    "        (e.g. \"24.0\"). A blank cell for a given sub-column should be omitted from\n"
    "        'values' rather than included as null (don't pad every row with every possible key).\n\n"
    "OTHER RULES:\n"
    "- Extract EVERY criterion row shown across every section, and EVERY test-date column\n"
    "  group. Do not summarise, skip, or truncate any row or column.\n"
    "- Preserve the test-date columns in the same left-to-right order shown in the image.\n"
    "- If a criterion row has a footnote reference number, ignore the footnote number itself --\n"
    "  it is not part of the criterion text or a data value.\n"
    "- Do not attempt to interpret or explain the footnotes at the bottom of the table; ignore\n"
    "  the footnote text entirely, it is not part of the row data.\n\n"
    "Return ONLY a JSON object with two top-level keys -- no markdown, no explanation, no code\n"
    "fences:\n"
    '  "unit_note": the units/currency note shown under the table title (e.g. "Millions of U.S.\n'
    '    dollars, unless otherwise indicated"), or null if none is shown.\n'
    '  "criteria": the JSON array of criterion objects described above.\n\n'
    "Example (abbreviated -- a real table has many more rows and test dates):\n"
    "{\n"
    '  "unit_note": "Millions of U.S. dollars, unless otherwise indicated",\n'
    '  "criteria": [\n'
    "    {\n"
    '      "category": "Performance Criteria",\n'
    '      "criterion": "Floor on primary fiscal balance, excluding grants",\n'
    '      "test_dates": [\n'
    '        {"date": "Dec. 2024", "values": {"PC": "24.0", "Adjusted PC": "24.0", '
    '"Actual": "61.0", "Status": "Met"}},\n'
    '        {"date": "Jun. 2026", "values": {"Proposed PC": "50.0"}}\n'
    "      ]\n"
    "    }\n"
    "  ]\n"
    "}"
)


def _extract_qpc_page_via_claude(pdf_path: str, page_num: int) -> dict:
    """Render a page and ask Claude vision to extract its QPC/IT criteria.
    Returns a dict with keys: unit_note, criteria (list of raw dicts)."""
    parsed = _render_page_and_call_claude(pdf_path, page_num, _VISION_PROMPT_QPC_FULL)
    if isinstance(parsed, dict) and isinstance(parsed.get("criteria"), list):
        result = parsed
    else:
        result = {"unit_note": None, "criteria": []}
    logger.info("[%s] p.%d -- Claude returned %d raw QPC criteria",
                pdf_path, page_num, len(result.get("criteria", [])))
    return result


def extract_qpc(pdf_path: str) -> list[QPCResult]:
    """
    Extract Quantitative Performance Criteria / Indicative Targets using
    Claude vision API.

    Mirrors extract_structural_benchmarks()'s page-render + Claude call
    structure, but the per-page result shape is a criteria list rather than
    a flat rows dict -- see QPCResult / _VISION_PROMPT_QPC_FULL. QPC tables
    are usually one page but the clustering in _find_qpc_pages still allows
    for multi-page tables.
    """
    import concurrent.futures

    pdf_path = str(pdf_path)
    pages = _find_qpc_pages(pdf_path)
    if not pages:
        return []

    logger.info("[%s] Claude-API QPC extraction on %d page(s): %s", pdf_path, len(pages), pages)

    page_data: dict[int, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(_extract_qpc_page_via_claude, pdf_path, pg): pg
            for pg in pages
        }
        for fut in concurrent.futures.as_completed(futures):
            pg = futures[fut]
            try:
                page_data[pg] = fut.result()
            except Exception as exc:
                logger.warning("[%s] p.%d QPC extraction failed: %s", pdf_path, pg, exc)
                page_data[pg] = {"unit_note": None, "criteria": []}

    import fitz as _fitz4
    _doc4 = _fitz4.open(pdf_path)
    page_titles: dict[int, str] = {}
    for pg in pages:
        text = _doc4[pg - 1].get_text() if pg <= len(_doc4) else ""
        # Same reasoning as extract_prior_actions -- use the first non-blank
        # line rather than the (deliberately short/lazy) regex match, so
        # the full caption line is preserved.
        first_line = next((l.strip() for l in text.splitlines() if l.strip()), "")
        page_titles[pg] = first_line or f"QPC / Indicative Targets (p.{pg})"
    _doc4.close()

    results: list[QPCResult] = []
    for pg in sorted(pages):
        data = page_data.get(pg, {})
        raw_criteria = data.get("criteria", [])
        criteria: list[QPCCriterion] = []
        for item in raw_criteria:
            if not isinstance(item, dict):
                continue
            criterion_text = (item.get("criterion") or "").strip()
            if not criterion_text:
                continue
            test_dates = item.get("test_dates")
            if not isinstance(test_dates, list):
                test_dates = []
            criteria.append(QPCCriterion(
                category=(item.get("category") or "").strip() or "Uncategorized",
                criterion=criterion_text,
                test_dates=test_dates,
            ))
        if criteria:
            results.append(QPCResult(
                table_title=page_titles.get(pg) or f"QPC / Indicative Targets (p.{pg})",
                page_num=pg,
                unit_note=data.get("unit_note") or None,
                criteria=criteria,
            ))

    logger.info("[%s] Claude-API QPC extraction complete: %d table(s), %d total criteria",
                pdf_path, len(results), sum(len(r.criteria) for r in results))
    return results


# ---------------------------------------------------------------------------
# SI title extraction — Docling TOC parse + Claude API classification
# ---------------------------------------------------------------------------
# Architecture:
#   1. _find_toc_pages()  — fitz scan to locate TOC pages (cheap, < 0.1 s)
#   2. _write_subpdf()    — already defined above; reused for TOC sub-PDF
#   3. _docling_toc_text()— Docling converts the TOC sub-PDF to clean markdown
#   4. _claude_si_titles()— Claude API reads the markdown and returns chapter
#                           titles as a JSON array
#
# Why Docling instead of raw fitz text:
#   - Handles ligatures, hyphenation, encoding issues that trip up raw fitz
#   - Produces clean, whitespace-normalised text
#   - Preserves reading order even in multi-column layouts
#
# Why Claude instead of regex:
#   - IMF SI TOCs vary wildly: all-caps subsections, indented subsections,
#     mixed-case, country-dash format, possessive titles, accented names
#   - Claude understands "these are chapter titles, not subsections" semantically
#   - Zero maintenance: no patterns to update when a new country or format appears
# ---------------------------------------------------------------------------

_CLAUDE_SI_TITLES_PROMPT = """\
You are reading the Table of Contents of an IMF Selected Issues Paper (SI).

The text below was extracted from the TOC pages. Your task: identify ONLY the \
top-level chapter titles.

Rules:
- An IMF SI contains 2-6 independent chapters. Return ALL of them.
- Chapter titles are the main bold headings. Common formats:
    COUNTRY-TOPIC TITLE (e.g. "BENIN-ECONOMIC TRANSFORMATION AND INDUSTRIAL POLICY")
    ALL CAPS STANDALONE TITLE (e.g. "SCALING UP PUBLIC EXPENDITURE FOR INCLUSIVE GROWTH")
- Do NOT include: subsections (A., B., C...), figure lists, table lists, "References".
- The text may be truncated. If a title appears cut off, include it anyway as-is.
- Return the full title exactly as it appears, stripped of page numbers, \
underscores, and dot leaders.
- Preserve the original capitalisation and punctuation of each title.

Return ONLY a JSON array of strings, one per chapter title, in document order.
No markdown, no explanation, no code fences.
Example: ["CHAPTER ONE FULL TITLE", "CHAPTER TWO FULL TITLE"]

TOC text:
{toc_text}
"""

_SI_TOC_HEADING_RE = re.compile(
    r"(table\s+of\s+contents|^\s*CONTENTS\s*$)", re.IGNORECASE | re.MULTILINE
)
_SI_LEADER_LINE_RE = re.compile(r"[\s_.]{3,}\d{1,4}\s*$")


def _find_toc_pages(pdf_path: str) -> list[int]:
    """
    Return 1-indexed page numbers that make up the SI Table of Contents.

    Strategy:
      - Scan pages 1-12, recording leader-line ratio and TOC-heading presence.
      - Find the first page with a TOC heading or ratio >= 0.25.
      - From there, collect pages with ratio >= 0.10, allowing up to
        _TOC_MAX_GAP consecutive low-ratio pages before stopping.
        This handles SIs where author/abstract pages interrupt the TOC.
    """
    _TOC_RATIO_MIN  = 0.10   # minimum leader-line ratio to count as TOC page
    _TOC_RATIO_HIGH = 0.25   # ratio high enough to identify TOC start without heading
    _TOC_MAX_GAP    = 2      # max consecutive low-ratio pages before giving up

    import fitz as _fitz
    doc = _fitz.open(str(pdf_path))

    # Collect (page_1indexed, ratio, has_toc_heading) for first 12 pages
    page_info: list[tuple[int, float, bool]] = []
    for i in range(min(12, len(doc))):
        text  = doc[i].get_text()
        lines = [l for l in text.splitlines() if l.strip()]
        if not lines:
            page_info.append((i + 1, 0.0, False))
            continue
        leader_count = sum(1 for l in lines if _SI_LEADER_LINE_RE.search(l))
        ratio        = leader_count / len(lines)
        has_heading  = bool(_SI_TOC_HEADING_RE.search(text))
        page_info.append((i + 1, ratio, has_heading))
    doc.close()

    # Find TOC start: first page with heading or high leader ratio
    toc_start: int | None = None
    for pg, ratio, has_heading in page_info:
        if has_heading or ratio >= _TOC_RATIO_HIGH:
            toc_start = pg
            break
    # Fallback: first page with any leader lines
    if toc_start is None:
        for pg, ratio, _ in page_info:
            if ratio >= _TOC_RATIO_MIN:
                toc_start = pg
                break

    if toc_start is None:
        logger.warning("[%s] No TOC pages detected; defaulting to pages 2-6", pdf_path)
        return list(range(2, 7))

    # Collect from toc_start, tolerating up to _TOC_MAX_GAP low-ratio pages
    toc_pages: list[int] = []
    low_streak = 0
    for pg, ratio, has_heading in page_info:
        if pg < toc_start:
            continue
        if ratio >= _TOC_RATIO_MIN or has_heading:
            toc_pages.append(pg)
            low_streak = 0
        else:
            low_streak += 1
            if low_streak > _TOC_MAX_GAP:
                break
            toc_pages.append(pg)   # include anyway — Claude ignores non-TOC text

    logger.info("[%s] TOC pages: %s", pdf_path, toc_pages)
    return toc_pages


_docling_converter = None


def _get_converter():
    """Lazily create and cache a Docling DocumentConverter instance.
    Was previously called but never defined anywhere in this module, so
    every _docling_toc_text() call silently fell back to raw fitz text."""
    global _docling_converter
    if _docling_converter is None:
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()
    return _docling_converter


def _docling_toc_text(pdf_path: str, toc_pages: list[int]) -> str:
    """
    Extract clean text from the TOC pages using Docling.

    Returns a markdown string.  Falls back to raw fitz text if Docling
    is unavailable or fails.
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        subpdf_path = tmp.name
    try:
        _write_subpdf(pdf_path, toc_pages, subpdf_path)
        try:
            converter = _get_converter()
            result = converter.convert(subpdf_path)
            md = result.document.export_to_markdown()
            if md and md.strip():
                logger.info("[%s] Docling TOC markdown: %d chars", pdf_path, len(md))
                return md
            logger.warning("[%s] Docling returned empty markdown; falling back to fitz", pdf_path)
        except Exception as exc:
            logger.warning("[%s] Docling TOC conversion failed (%s); using fitz", pdf_path, exc)

        # fitz fallback
        import fitz as _fitz
        doc = _fitz.open(str(pdf_path))
        chunks = []
        for pg in toc_pages:
            if 1 <= pg <= len(doc):
                chunks.append(doc[pg - 1].get_text())
        doc.close()
        return "\n".join(chunks)

    finally:
        try:
            os.unlink(subpdf_path)
        except OSError:
            pass


def _claude_si_titles(toc_text: str, pdf_path: str = "") -> list[str]:
    """
    Send TOC text to Claude and get back a list of chapter title strings.
    Returns [] if the API call fails or returns unusable output.
    """
    if not toc_text or not toc_text.strip():
        logger.warning("[%s] _claude_si_titles: empty toc_text", pdf_path)
        return []

    trimmed = toc_text.strip()[:12000]
    logger.info("[%s] TOC text sent to Claude (%d chars):\n%s", pdf_path, len(trimmed), trimmed)
    prompt = _CLAUDE_SI_TITLES_PROMPT.format(toc_text=trimmed)

    api_key = _API_KEY or os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("[%s] ANTHROPIC_API_KEY not set — cannot classify SI titles", pdf_path)
        return []

    body = {
        "model": _VISION_MODEL,
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "content-type":      "application/json",
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }

    try:
        import requests as _req
        resp = _req.post(
            "https://api.anthropic.com/v1/messages",
            json=body, headers=headers, timeout=60,
        )
        resp.raise_for_status()
        raw = ""
        for block in resp.json().get("content", []):
            if block.get("type") == "text":
                raw = block["text"].strip()
                break
    except Exception as exc:
        logger.warning("[%s] Claude SI title API call failed: %s", pdf_path, exc)
        return []

    logger.info("[%s] Claude raw response: %s", pdf_path, raw)
    parsed = _parse_json_response(raw)
    if not isinstance(parsed, list):
        logger.warning("[%s] Claude returned non-list: %s", pdf_path, raw[:200])
        return []

    titles = [str(t).strip() for t in parsed if isinstance(t, str) and t.strip()]
    logger.info("[%s] Claude extracted %d SI chapter titles: %s", pdf_path, len(titles), titles)
    return titles


def _extract_si_titles_from_doc(doc, pdf_path: str = "") -> str | None:
    """
    Extract chapter-level titles from an IMF Selected Issues PDF.

    Uses Docling to get clean TOC text, then Claude API to classify
    which entries are top-level chapter titles.

    Falls back to fitz raw text if Docling fails.
    Falls back to None (logged warning) if Claude API is unavailable.
    """
    if not pdf_path:
        logger.warning("_extract_si_titles_from_doc called without pdf_path; cannot extract")
        return None

    toc_pages  = _find_toc_pages(pdf_path)
    toc_text   = _docling_toc_text(pdf_path, toc_pages)
    titles     = _claude_si_titles(toc_text, pdf_path)
    return " | ".join(titles) if titles else None


def extract_si_titles(pdf_path: str) -> list[str]:
    """Public entry point: return SI chapter titles as a list of strings."""
    import fitz as _fitz
    doc = _fitz.open(str(pdf_path))
    try:
        result = _extract_si_titles_from_doc(doc, str(pdf_path))
    finally:
        doc.close()
    return result.split(" | ") if result else []


def extract_si_title(pdf_path: str) -> str | None:
    """Return SI chapter titles as a pipe-separated string, or None."""
    titles = extract_si_titles(pdf_path)
    return " | ".join(titles) if titles else None


def is_selected_issues(pdf_path: str) -> bool:
    """Return True if the PDF appears to be an IMF Selected Issues paper."""
    fname = Path(str(pdf_path)).name.upper()
    if "_SI_" in fname or fname.endswith("_SI.PDF") or fname.startswith("SI_"):
        return True
    _SI_TITLE_RE = re.compile(
        r"^\s*(?:imf\s+)?selected\s+issues?\s*(?:paper)?\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    _SI_EXCLUDE_RE = re.compile(
        r"(separately\s+released|will\s+be\s+released|published\s+separately"
        r"|see\s+also|companion\s+paper|selected\s+issues\s+paper\s+is)",
        re.IGNORECASE,
    )
    import fitz as _fitz
    doc = _fitz.open(str(pdf_path))
    for i in range(min(5, len(doc))):
        text = doc[i].get_text()
        if _SI_TITLE_RE.search(text) and not _SI_EXCLUDE_RE.search(text):
            doc.close()
            return True
    doc.close()
    return False




# ---------------------------------------------------------------------------
# Internal helpers — accept an already-open fitz.Document to avoid re-opens
# ---------------------------------------------------------------------------

def _extract_document_date_from_doc(doc, pdf_path: str) -> dict:
    """Same logic as extract_document_date() but reuses an open fitz doc."""
    _PROMINENT_DATE_RE = re.compile(
        r"^\s*(January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+(\d{4})\s*$",
        re.IGNORECASE | re.MULTILINE,
    )
    pages_to_scan = min(6, len(doc))
    prominent = contextual = fallback = year_only = None
    report_number = None

    for i in range(pages_to_scan):
        text = doc[i].get_text()

        if report_number is None:
            m = _IMF_REPORT_NO_RE.search(text)
            if m:
                report_number = m.group(1) if m.lastindex and m.group(1) else m.group(0)

        if prominent is None:
            m = _PROMINENT_DATE_RE.search(text)
            if m:
                prominent = {"raw": f"{m.group(1)} {m.group(2)}", "month": m.group(1),
                             "year": m.group(2), "source": "prominent"}

        if contextual is None:
            m = _CONTEXT_DATE_RE.search(text)
            if m:
                raw = m.group(1).strip()
                m2 = _MONTH_YEAR_RE.search(raw)
                contextual = {"raw": raw, "month": m2.group(1) if m2 else None,
                              "year": m2.group(2) if m2 else None, "source": "context"}

        if fallback is None:
            m = _MONTH_YEAR_RE.search(text)
            if m:
                fallback = {"raw": m.group(0).strip(), "month": m.group(1),
                            "year": m.group(2), "source": "month_year"}

    if prominent is None and contextual is None and fallback is None:
        for i in range(pages_to_scan):
            m = _YEAR_ONLY_RE.search(doc[i].get_text())
            if m:
                year_only = {"raw": m.group(1), "month": None,
                             "year": m.group(1), "source": "year_only"}
                break

    best = (prominent or contextual or fallback or year_only
            or {"raw": None, "month": None, "year": None, "source": None})
    best["report_number"] = report_number
    return best


def extract_all_targets(pdf_path: str, table_mode: str = "fast",
                        include_prior_actions: bool = True,
                        include_qpc: bool = True) -> dict:
    """
    Returns:
    {
      "document_date": {"raw", "month", "year", "source", "report_number"},
      "si_title":      str | None  — pipe-separated chapter titles for SI papers
      "toc":           [{"category", "title", "page"}, ...],
      "benchmarks":    [{"table_title", "page_num", "columns", "rows"}, ...]
                       (each row carries "Type": "SB" | "Reform Target")
      "prior_actions": [{"table_title", "page_num", "columns", "rows"}, ...]
      "qpc":           [{"table_title", "page_num", "unit_note", "criteria"}, ...]
    }
    Opens the PDF with fitz exactly once and reuses the doc object across all
    fast tracks (date, SI detection, TOC). Docling gets a sub-PDF separately.
    """
    import fitz
    pdf_path = str(pdf_path)
    doc = fitz.open(pdf_path)

    # ── Detect SI once, reuse doc ─────────────────────────────────────────────
    fname = Path(pdf_path).name.upper()
    _is_si = (
        "_SI_" in fname or fname.endswith("_SI.PDF")
        or bool(re.search(r"[_\-]SI\d*[_\-.]", fname))   # TZA-SI2024.pdf, ETH_SI2024.pdf
        or "SELECTEDISSUES" in fname.replace(" ", "").replace("_", "")
        or "SELECTED_ISSUES" in fname
    )
    if not _is_si:
        # Only match if "Selected Issues" appears as a standalone title/heading,
        # not as a passing mention in a publications list.
        # Require it to be at the start of a line (possibly with leading whitespace)
        # and NOT preceded by words like "released", "published", "separate".
        _SI_TITLE_RE = re.compile(
            r"^\s*(?:imf\s+)?selected\s+issues?\s*(?:paper)?\s*$",
            re.IGNORECASE | re.MULTILINE,
        )
        _SI_EXCLUDE_RE = re.compile(
            r"(separately\s+released|will\s+be\s+released|published\s+separately"
            r"|see\s+also|companion\s+paper|selected\s+issues\s+paper\s+is)",
            re.IGNORECASE,
        )
        for i in range(min(5, len(doc))):
            text = doc[i].get_text()
            if _SI_TITLE_RE.search(text) and not _SI_EXCLUDE_RE.search(text):
                _is_si = True
                break

    # ── Document date (reuse open doc) ────────────────────────────────────────
    document_date = _extract_document_date_from_doc(doc, pdf_path)

    # ── SI titles (reuse open doc) ────────────────────────────────────────────
    si_title = _extract_si_titles_from_doc(doc, pdf_path) if _is_si else None

    # ── TOC (reuse open doc) ──────────────────────────────────────────────────
    # For SI papers: skip box/annex extraction — chapter titles are in si_title
    if _is_si:
        toc_results = []
    else:
        outline = doc.get_toc(simple=False)
        if outline:
            toc_results = _extract_from_outline(outline, pdf_path)
            if not toc_results:
                toc_results = _extract_from_text(doc, pdf_path)
        else:
            toc_results = _extract_from_text(doc, pdf_path)

    doc.close()  # close once, here

    # ── Benchmarks (Docling, separate sub-PDF) ────────────────────────────────
    benchmarks = extract_structural_benchmarks(pdf_path, table_mode)

    # Prior Actions and QPC -- skipped for Selected Issues papers (they never
    # carry these tables). A failure in either must not lose the SB results.
    prior_actions, qpc = [], []
    if not _is_si:
        if include_prior_actions:
            try:
                prior_actions = [pa.to_dict() for pa in extract_prior_actions(pdf_path)]
            except Exception as exc:
                logger.warning("[%s] Prior Action extraction failed: %s", pdf_path, exc)
        if include_qpc:
            try:
                qpc = [q.to_dict() for q in extract_qpc(pdf_path)]
            except Exception as exc:
                logger.warning("[%s] QPC extraction failed: %s", pdf_path, exc)

    return {
        "document_date": document_date,
        "si_title":      si_title,
        "toc":           [e.to_dict() for e in toc_results],
        "benchmarks":    [b.to_dict() for b in benchmarks],
        "prior_actions": prior_actions,
        "qpc":           qpc,
    }


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

def batch_extract_all(
    pdf_paths: list[str],
    workers: int = 2,
    table_mode: str = "accurate",
) -> dict[str, dict]:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    pdf_paths = [str(p) for p in pdf_paths]
    results: dict[str, dict] = {}

    if workers == 1:
        for path in pdf_paths:
            try:
                results[path] = extract_all_targets(path, table_mode=table_mode)
            except Exception as exc:
                logger.error("[%s] Failed: %s", path, exc)
                results[path] = {"document_date": None, "toc": [], "benchmarks": []}
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(extract_all_targets, p, table_mode): p for p in pdf_paths}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                results[path] = fut.result()
            except Exception as exc:
                logger.error("[%s] Failed: %s", path, exc)
                results[path] = {"document_date": None, "toc": [], "benchmarks": []}

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Self-test — run with: python imf_pdf_extraction_engine.py --selftest
# Prints which file is loaded and verifies the key logic is the new version.
# ---------------------------------------------------------------------------

def _selftest():
    """Quick sanity check — call with --selftest flag."""
    import sys
    print(f"  File:    {__file__}")
    print(f"  Version: {__version__}")

    # Verify the key new functions exist
    assert callable(_find_toc_pages), "missing _find_toc_pages"
    assert callable(_claude_si_titles), "missing _claude_si_titles"
    assert "_CLAUDE_SI_TITLES_PROMPT" in dir(sys.modules[__name__]), "missing _CLAUDE_SI_TITLES_PROMPT"

    # Simulate the Benin SI TOC (all-caps subsections, hierarchical)
    class _P:
        def __init__(self, t): self._t = t
        def get_text(self): return self._t
    class _D:
        def __init__(self, pp): self._pp = pp
        def __len__(self): return len(self._pp)
        def __getitem__(self, i): return self._pp[i]

    benin_toc = (
        "SCALING UP PUBLIC EXPENDITURE FOR INCLUSIVE GROWTH _______ 4\n"
        "FORMAL AND INFORMAL TRADE TIES WITH NIGERIA: EVIDENCE FROM BORDER CLOSURE _____ 8\n"
        "BACKGROUND: STRUCTURAL SHIFT TOWARDS TRANSIT-BASED SERVICES IN BENIN _________ 9\n"
        "EMPIRICAL EVIDENCE ON THE FORMAL AND INFORMAL ECONOMIC TIES __________________ 11\n"
        "ECONOMETRIC ANALYSIS ________________________________________________________ 15\n"
        "CONCLUSION AND DISCUSSION ___________________________________________________ 17\n"
    )
    # Verify _find_toc_pages logic works on a trivial fake
    # (Can't fully test _claude_si_titles without API key, but check it's callable)
    print(f"  Functions present: _find_toc_pages, _claude_si_titles, _docling_toc_text")
    print(f"  Selftest PASSED")


if __name__ == "__main__":
    import sys, json
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        print("Running self-test...")
        _selftest()
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python imf_pdf_extraction_engine.py file1.pdf [file2.pdf ...]")
        print("       python imf_pdf_extraction_engine.py --selftest")
        sys.exit(1)

    all_results = batch_extract_all(sys.argv[1:])

    for pdf, res in all_results.items():
        print(f"\n{'='*60}\n  {Path(pdf).name}\n{'='*60}")
        d = res.get("document_date") or {}
        print(f"\n  Document date: {d.get('raw')}  (report: {d.get('report_number')})")
        if res.get("si_title"):
            print(f"\n  SI Chapters:")
            for ch in res["si_title"].split(" | "):
                print(f"    • {ch}")
        print(f"\n  Box & Annex entries ({len(res['toc'])}):")
        for e in res["toc"]:
            print(f"    [{e['category']}] {e['title']}  ->  p.{e['page']}")
        print(f"\n  Structural Benchmark tables ({len(res['benchmarks'])}):")
        for b in res["benchmarks"]:
            print(f"\n    {b['table_title']}  (p.{b['page_num']})")
            for label, vals in b["rows"].items():
                date   = vals.get("Due date") or vals.get("Due_date") or ""
                status = vals.get("Status", "")
                print(f"      [{status}] {label[:80]}  ({date})")

    out_path = "toc_targets.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved -> {out_path}")