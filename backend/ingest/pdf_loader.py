"""
pdf_loader.py — Structure-aware PDF loader with parent chunk builder.

Pipeline:
  load_pdf() → clean_text() → detect_sections() → build_parent_chunks()
"""

import re
from pathlib import Path
from typing import List, Tuple

import fitz  # PyMuPDF

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

PageDoc = dict      # raw page-level output from fitz
ParentChunk = dict  # logical section unit ready for child splitting

# ---------------------------------------------------------------------------
# Document type detection
# ---------------------------------------------------------------------------

DOC_TYPE_MAP = {
    "kust_statutes_2016":                              "statute",
    "kpuniversitesact2012":                            "act",
    "final revised semester rules 2019_260606_125443": "semester_rules",
    "hed_kp_harassment_policy_implementation":         "policy",
    "anomaliesofkust":                                 "amendment",
}

def _detect_doc_type(source: str) -> str:
    s = source.lower()
    for key, doc_type in DOC_TYPE_MAP.items():
        if key in s:
            return doc_type
    return "general"

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

PATTERNS = {
    "chapter":      re.compile(r'^CHAPTER[\s\-]*[IVXLC\d]+', re.MULTILINE | re.IGNORECASE),
    "schedule":     re.compile(r'^SCHEDULE', re.MULTILINE | re.IGNORECASE),
    "section_num":  re.compile(r'^\d+\.\s+\S', re.MULTILINE),
    "nested_rule":  re.compile(r'^\d+(?:\.\d+)+\s+\S', re.MULTILINE),
    "alpha_clause": re.compile(r'^\(?[a-z]\)\s+\S', re.MULTILINE),
    "roman_clause": re.compile(r'^\(?(?:i{1,3}|iv|vi{0,3}|ix|x)\)\s+', re.MULTILINE | re.IGNORECASE),
    "amendment":    re.compile(
        r'(for section\s+\d+|shall be substituted|shall be deleted|'
        r'shall be added|shall be inserted|after the word)',
        re.IGNORECASE
    ),
    "section_number_extract": re.compile(r'^(\d+(?:\.\d+)*)\s+'),
}

HARASSMENT_SECTIONS = [
    "General Guidelines", "Interaction Protocols",
    "Faculty", "Staff Responsibilities", "Campus Security",
    "Reporting and Response", "Training and Awareness",
    "Monitoring and Compliance", "Student Committees",
    "Review and Revision", "Implementation", "Acknowledgment",
]

# ---------------------------------------------------------------------------
# Step 1: Extract raw pages
# ---------------------------------------------------------------------------

def _extract_page_text(page: fitz.Page) -> str:
    blocks = page.get_text("blocks")
    text_blocks = sorted(
        [b for b in blocks if b[6] == 0],
        key=lambda b: (round(b[1] / 20), b[0])
    )
    return "\n".join(b[4].strip() for b in text_blocks if b[4].strip())


def load_pdf(file_path: Path) -> List[PageDoc]:
    """Extract text page by page from a single PDF."""
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    doc = fitz.open(str(file_path))
    source = file_path.stem
    total_pages = len(doc)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = _extract_page_text(page)
        if len(text.strip()) < 30:
            continue
        pages.append({
            "source":      source,
            "page":        page_num,
            "total_pages": total_pages,
            "text":        text,
        })

    doc.close()
    print(f"[pdf_loader] '{source}' — {len(pages)}/{total_pages} pages extracted")
    return pages


def load_all_pdfs(pdf_dir: Path) -> List[PageDoc]:
    """Recursively load all PDFs in a directory."""
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")

    pdf_files = list(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        print(f"[pdf_loader] No PDFs found in {pdf_dir}")
        return []

    print(f"[pdf_loader] Found {len(pdf_files)} PDF(s)")
    all_pages = []
    for path in pdf_files:
        try:
            all_pages.extend(load_pdf(path))
        except Exception as e:
            print(f"[pdf_loader] WARNING: Skipping '{path.name}' — {e}")

    print(f"[pdf_loader] Total pages extracted: {len(all_pages)}")
    return all_pages


# ---------------------------------------------------------------------------
# Step 2: Clean text
# ---------------------------------------------------------------------------

_NOISE_PATTERNS = [
    re.compile(r'^\d+\s*\|\s*Page\s*$',          re.MULTILINE | re.IGNORECASE),
    re.compile(r'^Page\s+\d+\s+of\s+\d+\s*$',    re.MULTILINE | re.IGNORECASE),
    re.compile(r'^KUST\s*$',                       re.MULTILINE),
    re.compile(r'^Kohat University.*$',            re.MULTILINE),
    re.compile(r'\f'),
]

def clean_text(text: str) -> str:
    for pattern in _NOISE_PATTERNS:
        text = pattern.sub('', text)

    text = re.sub(r'[ \t]{3,}', '  ', text)
    text = re.sub(r'\n{4,}', '\n\n\n', text)

    # Normalize unicode punctuation
    text = text.replace('\u2013', '-').replace('\u2014', '-')
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")

    return text.strip()


# ---------------------------------------------------------------------------
# Step 3: Section boundary detection
# ---------------------------------------------------------------------------

def _find_boundaries(text: str, doc_type: str) -> List[int]:
    """Return sorted character positions of section boundaries."""
    boundaries = set([0])

    if doc_type == "act":
        for m in PATTERNS["chapter"].finditer(text):
            boundaries.add(m.start())
        for m in PATTERNS["section_num"].finditer(text):
            boundaries.add(m.start())

    elif doc_type == "statute":
        for m in PATTERNS["chapter"].finditer(text):
            boundaries.add(m.start())
        for m in PATTERNS["section_num"].finditer(text):
            boundaries.add(m.start())
        for m in PATTERNS["nested_rule"].finditer(text):
            boundaries.add(m.start())

    elif doc_type == "semester_rules":
        for m in PATTERNS["nested_rule"].finditer(text):
            boundaries.add(m.start())
        for m in PATTERNS["section_num"].finditer(text):
            boundaries.add(m.start())

    elif doc_type == "policy":
        for title in HARASSMENT_SECTIONS:
            idx = text.find(title)
            if idx != -1:
                boundaries.add(idx)

    elif doc_type == "amendment":
        for m in PATTERNS["amendment"].finditer(text):
            line_start = text.rfind('\n', 0, m.start()) + 1
            boundaries.add(line_start)
        for m in PATTERNS["section_num"].finditer(text):
            boundaries.add(m.start())

    else:
        for m in PATTERNS["section_num"].finditer(text):
            boundaries.add(m.start())

    return sorted(boundaries)


def _extract_section_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:120]
    return "Untitled Section"


def _extract_section_number(text: str) -> str:
    """Extract leading section number like '15.2.7' or '5' from text."""
    m = PATTERNS["section_number_extract"].match(text.strip())
    return m.group(1) if m else ""


def _detect_amendment_metadata(text: str) -> dict:
    """
    For amendment chunks, extract what statute/section is being modified.
    Returns partial metadata dict.
    """
    meta = {
        "is_amendment":      True,
        "amends_source":     "KUST_Statutes_2016",
        "amends_section":    None,
        "amendment_action":  None,
    }

    # Try to find which section is being amended
    section_match = re.search(r'(?:for\s+)?section\s+(\d+[\w.]*)', text, re.IGNORECASE)
    if section_match:
        meta["amends_section"] = f"section {section_match.group(1)}"

    # Detect action type
    if re.search(r'shall be substituted', text, re.IGNORECASE):
        meta["amendment_action"] = "substituted"
    elif re.search(r'shall be deleted', text, re.IGNORECASE):
        meta["amendment_action"] = "deleted"
    elif re.search(r'shall be added', text, re.IGNORECASE):
        meta["amendment_action"] = "added"
    elif re.search(r'shall be inserted|after the word', text, re.IGNORECASE):
        meta["amendment_action"] = "inserted"

    return meta


# ---------------------------------------------------------------------------
# Step 4: Build parent chunks
# ---------------------------------------------------------------------------

def build_parent_chunks(pages: List[PageDoc]) -> List[ParentChunk]:
    """
    Convert raw pages into structured parent chunks.

    Each parent chunk = one complete logical section (statute section,
    act section, rule, policy clause, or amendment item).
    """
    if not pages:
        return []

    source = pages[0]["source"]
    doc_type = _detect_doc_type(source)
    total_pages = pages[0]["total_pages"]

    # Join all pages into one text, tracking page boundaries by char offset
    full_text = ""
    page_offsets: List[Tuple[int, int]] = []  # (char_start, page_number)

    for page_doc in pages:
        cleaned = clean_text(page_doc["text"])
        offset_start = len(full_text)
        full_text += cleaned + "\n\n"
        page_offsets.append((offset_start, page_doc["page"]))

    def _char_to_page(char_pos: int) -> int:
        """Map a character position back to its source page number."""
        page_num = page_offsets[0][1]
        for offset, pnum in page_offsets:
            if char_pos >= offset:
                page_num = pnum
            else:
                break
        return page_num

    # Detect section boundaries
    boundaries = _find_boundaries(full_text, doc_type)
    boundaries.append(len(full_text))  # sentinel end

    parents = []
    parent_index = 0

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end   = boundaries[i + 1]
        text  = full_text[start:end].strip()

        if len(text) < 40:  # skip tiny fragments (noise)
            continue

        section_num   = _extract_section_number(text)
        section_title = _extract_section_title(text)
        page_start    = _char_to_page(start)
        page_end      = _char_to_page(end)

        parent_id = f"{source}::section_{section_num or parent_index}"

        meta = {
            "parent_id":     parent_id,
            "source":        source,
            "doc_type":      doc_type,
            "title":         source.replace("_", " "),
            "section":       section_num,
            "section_title": section_title,
            "page_start":    page_start,
            "page_end":      page_end,
            "total_pages":   total_pages,
            "text":          text,
            "is_amendment":  False,
            "amends_source": None,
            "amends_section": None,
            "amendment_action": None,
        }

        # Enrich amendment metadata
        if doc_type == "amendment":
            meta.update(_detect_amendment_metadata(text))

        parents.append(meta)
        parent_index += 1

    print(f"[pdf_loader] '{source}' ({doc_type}) → {len(parents)} parent chunks")
    return parents


def build_all_parent_chunks(pdf_dir: Path) -> List[ParentChunk]:
    """Full pipeline: load all PDFs → build parent chunks for all."""
    all_pages = load_all_pdfs(pdf_dir)

    # Group pages by source document
    sources: dict[str, List[PageDoc]] = {}
    for page in all_pages:
        sources.setdefault(page["source"], []).append(page)

    all_parents = []
    for source, pages in sources.items():
        parents = build_parent_chunks(pages)
        all_parents.extend(parents)

    print(f"[pdf_loader] Total parent chunks across all docs: {len(all_parents)}")
    return all_parents