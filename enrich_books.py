"""
Automated Book & Media Library Metadata Enrichment Tool for ML & Media Management.

Queries public book APIs (Open Library, Crossref, Google Books) with rate limiting,
caching, and title similarity verification, combined with an extensive local rule-based
and heuristic taxonomy engine to populate a feature-rich ML dataset:

Schema Columns:
1.  id                  INTEGER PRIMARY KEY
2.  clean_title         TEXT (standard proper-cased title)
3.  subtitle            TEXT (extracted or API subtitle)
4.  raw_title           TEXT (original filename/entry)
5.  author              TEXT (author/creator names)
6.  publisher           TEXT (publisher / platform e.g. O'Reilly, Packt, Wiley, McGraw-Hill, Udemy)
7.  published           TEXT (publication year / date)
8.  edition             TEXT (edition string e.g. "3rd Edition", "1st Edition")
9.  revision            INTEGER (edition number integer e.g. 1, 2, 3)
10. language            TEXT (ISO language code e.g. "en")
11. domain              TEXT (broad domain e.g. "Computer Science & Software", "Engineering & Hardware")
12. subject             TEXT (specific field e.g. "Artificial Intelligence & Data Science")
13. subfield            TEXT (granular topic e.g. "Deep Learning & Neural Networks")
14. genre               TEXT (e.g. "Technical Reference", "Textbook", "Field Guide", "Courseware")
15. topics              TEXT (comma-separated keyword tags for embeddings / search)
16. key_concepts        TEXT (key principles / topics covered for ML tokenization & context)
17. target_audience     TEXT (e.g. "Beginner", "Intermediate", "Advanced", "Practitioner")
18. media_type          TEXT (e.g. "Book / Ebook", "Course / Video Pack", "Project Documentation", "Collection / Pack")
19. is_collection       INTEGER (1 if collection/pack, 0 otherwise)
20. summary             TEXT (rich descriptive abstract / synopsis for LLMs & semantic search)
21. file_type           TEXT (format e.g. "PDF", "EPUB", "DOCX", "Markdown", "Folder")
22. file_extension      TEXT (raw extension e.g. ".pdf", ".epub")
23. source_path         TEXT (path on disk)
24. in_directory        INTEGER (availability flag)
25. created_at          TIMESTAMP
26. updated_at          TIMESTAMP
"""

from __future__ import annotations

import argparse
import csv
import html
import io
import json
import os
import re
import sqlite3
import ssl
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from front_matter import (
    extract_front_matter,
    is_uninformative_title,
    query_online_by_isbn,
    FrontMatterResult,
)

DEFAULT_DB_PATH = Path("books.db")
DEFAULT_CSV_PATH = Path("books_db.csv")
DEFAULT_CACHE_PATH = Path(".enrichment_cache.json")
DEFAULT_DIRECTORY = Path("/Volumes/Drive2/Books_2/")

# SSL Context Setup with fallback
SSL_CONTEXT = ssl.create_default_context()
try:
    import certifi
    SSL_CONTEXT.load_verify_locations(certifi.where())
except Exception:
    SSL_CONTEXT = ssl._create_unverified_context()

USER_AGENT = "BookCatalogEnricher/1.0 (https://github.com/curedsounds/home_cc; contact@curedsounds.com)"

KNOWN_PUBLISHERS = [
    ("O'Reilly", re.compile(r"\b(o'?reilly|oreilly)\b", re.I)),
    ("Packt Publishing", re.compile(r"\b(packt|packtpub)\b", re.I)),
    ("Wiley", re.compile(r"\bwiley\b", re.I)),
    ("McGraw-Hill", re.compile(r"\b(mcgraw[- ]?hill)\b", re.I)),
    ("Manning Publications", re.compile(r"\bmanning\b", re.I)),
    ("Apress", re.compile(r"\bapress\b", re.I)),
    ("No Starch Press", re.compile(r"\b(no starch|nostarch)\b", re.I)),
    ("Addison-Wesley", re.compile(r"\b(addison[- ]?wesley)\b", re.I)),
    ("Prentice Hall", re.compile(r"\b(prentice[- ]?hall)\b", re.I)),
    ("MIT Press", re.compile(r"\bmit press\b", re.I)),
    ("Cambridge University Press", re.compile(r"\bcambridge\b", re.I)),
    ("Oxford University Press", re.compile(r"\boxford\b", re.I)),
    ("Springer", re.compile(r"\bspringer\b", re.I)),
    ("CRC Press", re.compile(r"\bcrc press\b", re.I)),
    ("Taylor & Francis", re.compile(r"\btaylor & francis\b", re.I)),
    ("Sybex", re.compile(r"\bsybex\b", re.I)),
    ("Udemy", re.compile(r"\budemy\b", re.I)),
    ("Coursera", re.compile(r"\bcoursera\b", re.I)),
    ("Black & Decker", re.compile(r"\bblack\s*(&|and)\s*decker\b", re.I)),
    ("Adafruit", re.compile(r"\badafruit\b", re.I)),
    ("SparkFun", re.compile(r"\bsparkfun\b", re.I)),
    ("Raspberry Pi Press", re.compile(r"\braspberry pi press\b", re.I)),
    ("Arduino LLC", re.compile(r"\barduino\b", re.I)),
]

NON_AUTHOR_TAGS = {
    "webtooltip.com",
    "freecourselab.com",
    "udemy",
    "coursera",
    "edx",
    "packt",
    "packtpub",
    "oreilly",
    "wiley",
    "manning",
    "apress",
    "mcgraw-hill",
    "mcgraw hill",
    "nostarch",
    "no starch press",
    "sybex",
    "rocketry",
    "electronics",
    "ebook",
    "pdf",
    "epub",
    "1080p",
    "720p",
    "tutorial",
    "chegg",
    "chegg solutions",
    "source files",
    "textbook",
    "solutions",
    "130+",
    "20",
    "30",
    "40",
    "150",
}

# Domain & Subject Hierarchy for ML Feature Classification
# Tuple: (RegexPattern, Domain, Subject, Subfield, Genre, DefaultAudience, KeyConcepts, Topics)
TAXONOMY_RULES: List[Tuple[re.Pattern, str, str, str, str, str, str, str]] = [
    # AI & Machine Learning
    (
        re.compile(r"\b(artificial\s+intelligence|agentic\s+ai|generative\s+ai|llm|deep\s+learning|machine\s+learning|neural\s+network|\bai\b|\bml\b|transformers?|gpt|langchain|reinforcement\s+learning|nlp|computer\s+vision|diffusion\s+models?)\b", re.I),
        "Computer Science & Artificial Intelligence",
        "Artificial Intelligence & Machine Learning",
        "Deep Learning & Generative AI",
        "Technical Reference",
        "Advanced / Practitioner",
        "Model Architectures, Training Pipelines, Neural Optimization, Embeddings, Agentic Workflows",
        "ai, machine-learning, deep-learning, llm, neural-networks, transformers, nlp, generative-ai",
    ),
    # Data Science & Analytics
    (
        re.compile(r"\b(data\s+science|data\s+scientist|pandas|numpy|scikit|analytics|big\s+data|data\s+mining|feature\s+engineering|time\s+series)\b", re.I),
        "Computer Science & Artificial Intelligence",
        "Data Science & Statistical Analysis",
        "Applied Data Science & Mining",
        "Technical Reference",
        "Intermediate / Practitioner",
        "Exploratory Data Analysis, Statistical Modeling, Feature Extraction, Data Pipelines",
        "data-science, python, statistics, analytics, machine-learning, data-engineering",
    ),
    # Algorithms & Software Engineering
    (
        re.compile(r"\b(algorithms?|data\s+structures?|software\s+architecture|design\s+patterns?|system\s+design|clean\s+code|refactoring|microservices|api\s+design)\b", re.I),
        "Computer Science & Artificial Intelligence",
        "Software Engineering & Computer Science",
        "Algorithms & System Architecture",
        "Textbook / Technical Reference",
        "Intermediate / Advanced",
        "Algorithmic Complexity, Big-O, Data Structures, Distributed Systems, Software Design",
        "algorithms, data-structures, computer-science, software-engineering, architecture",
    ),
    # Programming Languages & Web Development
    (
        re.compile(r"\b(python|javascript|typescript|c\+\+|golang|rust|java|react|vue|node\.?js|html|css|php|backend|frontend|fullstack|web\s+development)\b", re.I),
        "Computer Science & Software Development",
        "Programming & Software Development",
        "Application & Web Engineering",
        "Technical Guide",
        "Beginner / Intermediate",
        "Syntax, Idiomatic Programming, Frameworks, Application Design, API Integration",
        "programming, software-development, web-development, python, coding, applications",
    ),
    # Cybersecurity & Hacking
    (
        re.compile(r"\b(cybersecurity|pentest|pentesting|ethical\s+hacking|malware|reverse\s+engineering|cryptography|cryptanalysis|security|wireshark|metasploit|vulnerability|network\s+defense|steganography)\b", re.I),
        "Information Security & Networks",
        "Cybersecurity & InfoSec",
        "Penetration Testing & Threat Analysis",
        "Technical Reference",
        "Intermediate / Advanced",
        "Vulnerability Assessment, Exploitation, Network Hardening, Threat Modeling, Cryptography",
        "cybersecurity, infosec, ethical-hacking, pentesting, network-security, reverse-engineering",
    ),
    # DevOps, Linux & Cloud Infrastructure
    (
        re.compile(r"\b(linux|ubuntu|debian|docker|kubernetes|aws|cloud|devops|ansible|terraform|bash|powershell|ci\/cd|server|sysadmin)\b", re.I),
        "Information Security & Networks",
        "Cloud Infrastructure & DevOps",
        "Containerization & System Administration",
        "Technical Guide",
        "Intermediate / Practitioner",
        "Infrastructure as Code, Container Orchestration, CI/CD Pipelines, Linux Administration",
        "devops, cloud, linux, docker, kubernetes, sysadmin, aws, infrastructure",
    ),
    # Electrical Engineering, Electronics & Embedded Systems
    (
        re.compile(r"\b(electric\s+motors?|circuits?|electronics?|the\s+art\s+of\s+electronics|digital\s+fundamentals|microcontroller|arduino|raspberry\s+pi|teensy|esp32|esp8266|stm32|arm|fpga|pcb|sensors?|semiconductors?|dsp|rf|vhdl|verilog|power\s+electronics)\b", re.I),
        "Engineering & Applied Physics",
        "Electrical & Embedded Engineering",
        "Circuit Design, Embedded Systems & Hardware",
        "Textbook / Technical Reference",
        "Intermediate / Practitioner",
        "Analog & Digital Circuits, Microcontrollers, Signal Processing, Sensor Interfacing, Hardware Design",
        "electronics, electrical-engineering, embedded-systems, arduino, raspberry-pi, circuits, sensors, pcb",
    ),
    # Mechanical Engineering, Lathe, Machining & Manufacturing
    (
        re.compile(r"\b(lathe|milling|machining|cnc|metalworking|blacksmithing|woodworking|carpentry|welding|sheet\s+metal|manufacturing|solidworks|autocad|cad|cam|workshop|mechanics)\b", re.I),
        "Engineering & Applied Physics",
        "Mechanical & Manufacturing Engineering",
        "Machining, Fabrication & Toolmaking",
        "Practical Guide / Manual",
        "Practitioner / Craftsman",
        "Precision Machining, Tooling Geometry, Fabrication, Metal Lathe Operations, CAD/CAM Modeling",
        "mechanical-engineering, machining, lathe, metalworking, fabrication, cnc, manufacturing",
    ),
    # Aerospace, Propulsion & Rocketry
    (
        re.compile(r"\b(rocketry|liquid\s+fuel|rocket\s+engine|aerospace|aeronautics|avionics|propulsion|orbital\s+mechanics|ballistics|flight)\b", re.I),
        "Engineering & Applied Physics",
        "Aerospace & Aeronautical Engineering",
        "Rocket Propulsion & Orbital Systems",
        "Technical Reference",
        "Advanced / Practitioner",
        "Combustion Dynamics, Nozzle Design, Propellant Chemistry, Avionics, Orbital Trajectories",
        "aerospace, rocketry, propulsion, rocket-engines, aerodynamics, physics",
    ),
    # Chemistry & Chemical Engineering
    (
        re.compile(r"\b(chemistry|organic\s+chemistry|biochemistry|chemical|molecular|reactions?|spectroscopy|periodic\s+table|materials\s+science)\b", re.I),
        "Natural & Physical Sciences",
        "Chemistry & Chemical Sciences",
        "Organic & Applied Chemistry",
        "Textbook",
        "Academic / Practitioner",
        "Molecular Structures, Reaction Mechanisms, Stoichiometry, Thermodynamics, Synthesis",
        "chemistry, organic-chemistry, science, biochemistry, molecular-science",
    ),
    # Astronomy & Cosmology
    (
        re.compile(r"\b(astronomy|astrophysics|night\s+sky|stargazing|telescope|cosmos|universe|planetary|planets|stars|galaxy|cosmology)\b", re.I),
        "Natural & Physical Sciences",
        "Astronomy & Space Science",
        "Observational Astronomy & Stargazing",
        "Field Guide / Educational",
        "General / Enthusiast",
        "Celestial Navigation, Constellations, Optics, Planetary Motion, Deep Sky Exploration",
        "astronomy, astrophysics, stargazing, night-sky, space, cosmos, telescopes",
    ),
    # Physics & Applied Physics
    (
        re.compile(r"\b(physics|quantum|thermodynamics|optics|electromagnetism|fluid\s+dynamics|statistical\s+physics|relativity)\b", re.I),
        "Natural & Physical Sciences",
        "Physics & Theoretical Physics",
        "Classical & Modern Physics",
        "Textbook",
        "Academic / Advanced",
        "Classical Mechanics, Electrodynamics, Quantum States, Energy Transfer, Mathematical Physics",
        "physics, quantum-mechanics, thermodynamics, electromagnetism, science",
    ),
    # Mathematics & Statistics
    (
        re.compile(r"\b(math|mathematics|calculus|linear\s+algebra|differential\s+equations|geometry|probability|statistics|discrete\s+math|number\s+theory)\b", re.I),
        "Formal Sciences & Mathematics",
        "Mathematics & Applied Statistics",
        "Foundational & Advanced Mathematics",
        "Textbook",
        "Academic / Practitioner",
        "Vector Spaces, Differential Operators, Probability Distributions, Proof Methods, Optimization",
        "mathematics, statistics, calculus, linear-algebra, probability, math",
    ),
    # Survival, Bushcraft & Wilderness Skills
    (
        re.compile(r"\b(survival|bushcraft|foraging|edible\s+wild|wilderness|prepper|preppers|first\s+aid|trauma|emergency|shelter|firemaking|cordage|hunting|fishing|trapping|orienteering|navigation|off\s+grid)\b", re.I),
        "Outdoor, Survival & Environmental Skills",
        "Survival & Bushcraft",
        "Wilderness Survival & Self-Reliance",
        "Field Guide / Manual",
        "Practitioner / Enthusiast",
        "Wilderness Medicine, Shelter Construction, Foraging & Identification, Navigation, Emergency Preparedness",
        "survival, bushcraft, wilderness, foraging, prepping, first-aid, self-reliance",
    ),
    # Gardening, Agriculture & Homesteading
    (
        re.compile(r"\b(gardening|greenhouse|permaculture|homesteading|beekeeping|apiculture|husbandry|soil|compost|plants|vegetables|hydroponics|aquaponics|orchard|backyard)\b", re.I),
        "Agriculture & Homesteading",
        "Horticulture & Permaculture",
        "Sustainable Agriculture & Homesteading",
        "Practical Guide / Manual",
        "Practitioner / Enthusiast",
        "Soil Ecology, Plant Cultivation, Greenhouse Systems, Crop Rotation, Animal Husbandry",
        "gardening, homesteading, permaculture, agriculture, greenhouse, beekeeping",
    ),
    # Home Improvement, Construction & Trades
    (
        re.compile(r"\b(home\s+improvement|plumbing|wiring|electrical\s+wiring|masonry|stonework|deck|shed|roofing|concrete|home\s+repair|bathrooms?|remodel|black\s*&\s*decker)\b", re.I),
        "Trades, Construction & Home Craft",
        "Construction & Home Improvement",
        "Residential Trades & Remodeling",
        "Practical Guide / Manual",
        "Homeowner / Craftsman",
        "Residential Wiring, Plumbing Diagnostics, Structural Carpentry, Masonry, Code Compliance",
        "home-improvement, construction, plumbing, electrical-wiring, carpentry, remodeling",
    ),
    # Audio Engineering, Synthesizers & Electronic Music
    (
        re.compile(r"\b(synthesizer|synthesizers|synth|modular\s+synth|eurorack|midi|audio\s+engineering|sound\s+design|music\s+production|acoustics|recording|mixing|mastering)\b", re.I),
        "Creative Arts & Music Technology",
        "Audio Engineering & Sound Design",
        "Synthesizers & Electronic Music Production",
        "Technical Reference",
        "Practitioner / Musician",
        "Subtractive/Additive Synthesis, MIDI Protocol, DSP Modulation, Acoustic Treatments, Mixing Signal Chains",
        "audio-engineering, synthesizers, music-production, sound-design, midi, acoustics",
    ),
    # Personal Development, Psychology & Philosophy
    (
        re.compile(r"\b(mind|organize|stoicism|habits|focus|psychology|productivity|cognitive|philosophy|epistemology|thinking|learning|meditation|clear\s+your\s+mind)\b", re.I),
        "Humanities & Social Sciences",
        "Personal Development & Philosophy",
        "Cognitive Strategy & Productivity",
        "Non-Fiction / Self-Improvement",
        "General / Practitioner",
        "Habit Architecture, Cognitive Reframing, Time Management, Stoic Principles, Focus Optimization",
        "personal-development, productivity, psychology, philosophy, habits, mindset",
    ),
    # Finance, Banking & Administration
    (
        re.compile(r"\b(statement|credit\s+card|invoice|tax|1040|bill|financial|accounting|ledger|banking|econometrics)\b", re.I),
        "Business, Finance & Administration",
        "Financial & Administrative Records",
        "Accounting, Taxes & Personal Finance",
        "Financial Document / Record",
        "Administrative / General",
        "Financial Reporting, Ledger Balances, Tax Schedules, Fiscal Accounting",
        "finance, accounting, taxes, financial-records, administrative",
    ),
    # System Architecture & Internal Manifests
    (
        re.compile(r"\b(manifest|diagnostic\s+ecosystem|cybernetic|architecture|system\s+specification|whitepaper)\b", re.I),
        "Systems Architecture & Engineering",
        "Systems Architecture & Ecosystems",
        "Cybernetic Diagnostics & Ecosystem Architecture",
        "Project Documentation",
        "Architect / Engineer",
        "Cybernetic Feedback Loops, Diagnostic Pipelines, Multi-Agent Instrumentation, System Design",
        "system-architecture, cybernetics, diagnostics, technical-specification, documentation",
    ),
]

EXTENSION_MAP: Dict[str, str] = {
    ".pdf": "PDF",
    ".epub": "EPUB",
    ".mobi": "MOBI",
    ".azw": "AZW",
    ".azw3": "AZW3",
    ".djvu": "DJVU",
    ".fb2": "FB2",
    ".ibooks": "iBooks",
    ".cbr": "CBR",
    ".cbz": "CBZ",
    ".txt": "Text",
    ".rtf": "RTF",
    ".doc": "Word",
    ".docx": "Word (DOCX)",
    ".odt": "ODT",
    ".md": "Markdown",
    ".markdown": "Markdown",
    ".zip": "ZIP Archive",
    ".rar": "RAR Archive",
    ".7z": "7Z Archive",
    ".tar": "TAR Archive",
    ".gz": "GZ Archive",
    ".csv": "CSV Spreadsheet",
    ".json": "JSON Data",
    ".py": "Python Script",
    ".html": "HTML Document",
    ".htm": "HTML Document",
}


@dataclass
class EnrichedBookRecord:
    clean_title: str
    subtitle: Optional[str] = None
    raw_title: str = ""
    author: Optional[str] = None
    publisher: Optional[str] = None
    published: Optional[str] = None
    edition: Optional[str] = None
    revision: int = 1
    language: str = "en"
    domain: str = "General Science & Technology"
    subject: str = "Technical & Applied Sciences"
    subfield: str = "General Technical Studies"
    genre: str = "Technical Reference"
    topics: str = ""
    key_concepts: str = ""
    target_audience: str = "General / Practitioner"
    media_type: str = "Book / Ebook"
    is_collection: int = 0
    summary: str = ""
    file_type: str = "Unknown"
    file_extension: str = ""
    source_path: Optional[str] = None
    in_directory: int = 0
    source: str = "local+rules"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clean_title": self.clean_title,
            "subtitle": self.subtitle,
            "raw_title": self.raw_title,
            "author": self.author,
            "publisher": self.publisher,
            "published": self.published,
            "edition": self.edition,
            "revision": self.revision,
            "language": self.language,
            "domain": self.domain,
            "subject": self.subject,
            "subfield": self.subfield,
            "genre": self.genre,
            "topics": self.topics,
            "key_concepts": self.key_concepts,
            "target_audience": self.target_audience,
            "media_type": self.media_type,
            "is_collection": self.is_collection,
            "summary": self.summary,
            "file_type": self.file_type,
            "file_extension": self.file_extension,
            "source_path": self.source_path,
            "in_directory": self.in_directory,
            "source": self.source,
        }


def clean_html_text(raw_html: Any) -> str:
    """Strips HTML tags and unescapes entities."""
    if not raw_html:
        return ""
    if isinstance(raw_html, (list, tuple)):
        raw_html = " ".join(str(item) for item in raw_html if item)
    elif isinstance(raw_html, dict):
        raw_html = raw_html.get("value", str(raw_html))
    text = re.sub(r"<[^>]+>", " ", str(raw_html))
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def calculate_title_similarity(s1: str, s2: str) -> float:
    """Calculates token Jaccard similarity between two title strings."""
    tokens1 = set(re.findall(r"\w+", s1.lower()))
    tokens2 = set(re.findall(r"\w+", s2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1.intersection(tokens2)
    union = tokens1.union(tokens2)
    return len(intersection) / len(union)


class MetadataCache:
    """Persistent JSON cache for online search queries."""

    def __init__(self, cache_file: Union[str, Path] = DEFAULT_CACHE_PATH):
        self.cache_file = Path(cache_file)
        self.data: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if self.cache_file.is_file():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except Exception:
                self.data = {}

    def save(self) -> None:
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self.data.get(key)

    def set(self, key: str, value: Dict[str, Any]) -> None:
        self.data[key] = value


class LocalRuleExtractor:
    """Extracts features, publishers, editions, taxonomy and semantics from titles."""

    @staticmethod
    def extract_file_info(raw_title: str, source_path: Optional[str] = None) -> Tuple[str, str]:
        # Check raw_title and source_path for file extensions
        candidates = [raw_title, source_path] if raw_title else [source_path]
        for name in candidates:
            if not name:
                continue
            # Strip emojis and leading/trailing whitespace
            clean_name = re.sub(r"^[📂📁📄\-_*•\s]+", "", name).strip()
            if clean_name.lower().endswith(".csv.docx"):
                return "Word (DOCX)", ".csv.docx"
            if clean_name.lower().endswith(".tar.gz"):
                return "TAR GZ Archive", ".tar.gz"

            p = Path(clean_name)
            suffix = p.suffix.lower()
            if suffix in EXTENSION_MAP:
                return EXTENSION_MAP[suffix], suffix
            if suffix and re.match(r"^\.[a-z0-9]{2,5}$", suffix):
                return suffix.replace(".", "").upper(), suffix

        return "Directory / Folder", ""

    @staticmethod
    def extract_publisher(raw_title: str, clean_title: str) -> Optional[str]:
        combined = f"{raw_title} {clean_title}"
        for pub_name, pattern in KNOWN_PUBLISHERS:
            if pattern.search(combined):
                return pub_name
        return None

    @staticmethod
    def extract_subtitle(clean_title: str) -> Tuple[str, Optional[str]]:
        """Splits title into main title and subtitle if colon or dash is used."""
        if ":" in clean_title:
            parts = clean_title.split(":", 1)
            return parts[0].strip(), parts[1].strip()
        if " - " in clean_title:
            parts = clean_title.split(" - ", 1)
            # Only treat second part as subtitle if first part is not a known prefix
            if parts[0].strip().lower() not in ["udemy", "statement", "2024"]:
                return parts[0].strip(), parts[1].strip()
        return clean_title, None

    @staticmethod
    def extract_edition_info(raw_title: str, clean_title: str) -> Tuple[int, str]:
        combined = f"{raw_title} {clean_title}".lower()
        # Pattern 1: 1st, 2nd, 3rd, 11th edition / ed
        m1 = re.search(r"\b(\d+)(?:st|nd|rd|th)\s+(?:edition|ed)\b", combined)
        if m1:
            rev = int(m1.group(1))
            suffix = "th" if 11 <= rev <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(rev % 10, "th")
            return rev, f"{rev}{suffix} Edition"

        # Pattern 2: pack-5, collection-3
        m3 = re.search(r"\bpack[-_\s]*(\d+)\b", combined)
        if m3:
            rev = int(m3.group(1))
            return rev, f"Pack {rev}"

        # Pattern 3: v7 / v2
        m4 = re.search(r"\bv(\d+)\b", combined)
        if m4:
            rev = int(m4.group(1))
            return rev, f"Version {rev}.0"

        return 1, "1st Edition"

    @staticmethod
    def extract_published_year(raw_title: str, clean_title: str) -> Optional[str]:
        # Dates like 2024-04-12
        date_match = re.search(r"\b(19\d\d|20\d\d)[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])\b", raw_title)
        if date_match:
            return date_match.group(0).replace("_", "-")

        # Parenthesized / bracketed years e.g. (2026), [2015]
        m = re.findall(r"[\(\[\s_-](19\d\d|20\d\d)[\)\]\s_.-]", f" {raw_title} ")
        if m:
            valid_years = [y for y in m if 1950 <= int(y) <= 2030]
            if valid_years:
                return valid_years[-1]

        # Check clean title
        m_clean = re.findall(r"\b(19\d\d|20\d\d)\b", clean_title)
        if m_clean:
            valid_years = [y for y in m_clean if 1950 <= int(y) <= 2030]
            if valid_years:
                return valid_years[-1]

        return None

    @staticmethod
    def extract_author(raw_title: str, clean_title: str) -> Optional[str]:
        # Bracket author at start
        m_bracket = re.match(r"^\s*\[([^\]]+)\]", raw_title)
        if m_bracket:
            candidate = m_bracket.group(1).strip()
            if candidate.lower() not in NON_AUTHOR_TAGS and not any(t in candidate.lower() for t in ["http", "www.", ".com", ".org"]):
                if not re.match(r"^\d+$", candidate):
                    return candidate

        # "by Author"
        m_by = re.search(r"\bby\s+([A-Z][a-z]+(?:\s+[A-Z][a-z\.]+){1,3})\b", clean_title)
        if m_by:
            return m_by.group(1).strip()

        # Known authoritative authors in textbooks & manuals
        known_authors = [
            ("Floyd", "Thomas L. Floyd"),
            ("Petruzella", "Frank D. Petruzella"),
            ("Horowitz", "Paul Horowitz & Winfield Hill"),
            ("Art of Electronics", "Paul Horowitz & Winfield Hill"),
            ("Sedra", "Adel S. Sedra & Kenneth C. Smith"),
            ("Boylestad", "Robert L. Boylestad"),
            ("Tanenbaum", "Andrew S. Tanenbaum"),
            ("Silberschatz", "Abraham Silberschatz"),
            ("Knuth", "Donald E. Knuth"),
            ("Cormen", "Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest"),
            ("Goodfellow", "Ian Goodfellow, Yoshua Bengio, Aaron Courville"),
            ("Geron", "Aurélien Géron"),
            ("Russell", "Stuart Russell & Peter Norvig"),
            ("Hoyle", "David Hoyle"),
            ("Ahmad", "Imran Ahmad"),
            ("Regas", "Dean Regas"),
            ("Canterbury", "Dave Canterbury"),
            ("Bushcraft 101", "Dave Canterbury"),
            ("Jose Portilla", "Jose Portilla"),
            ("Python for Data Science and Machine Learning", "Jose Portilla"),
            ("Liquid Fuel Rocket", "Leroy J. Krzycki"),
            ("Krzycki", "Leroy J. Krzycki"),
            ("James Stewart", "James Stewart"),
        ]
        for keyword, full_name in known_authors:
            if re.search(rf"\b{re.escape(keyword)}\b", clean_title, re.I) or re.search(rf"\b{re.escape(keyword)}\b", raw_title, re.I):
                return full_name

        return None

    @staticmethod
    def match_taxonomy(raw_title: str, clean_title: str) -> Tuple[str, str, str, str, str, str, str]:
        combined = f"{clean_title} {raw_title}".lower()
        for pattern, domain, subject, subfield, genre, audience, concepts, topics in TAXONOMY_RULES:
            if pattern.search(combined):
                return domain, subject, subfield, genre, audience, concepts, topics

        # Fallbacks
        return (
            "General Science & Technology",
            "Technical & Applied Sciences",
            "General Technical Studies",
            "Technical Reference",
            "General / Practitioner",
            "General Methodologies, Practical Applications, Core Concepts",
            "technology, reference, engineering, science",
        )

    @staticmethod
    def determine_media_classification(raw_title: str, clean_title: str, file_type: str) -> Tuple[str, int]:
        combined = f"{clean_title} {raw_title}".lower()
        is_collection = 0

        if "udemy" in combined or "bootcamp" in combined or "course" in combined or "tutorial" in combined:
            media_type = "Course / Video Series"
            is_collection = 1 if "bootcamp" in combined or "pack" in combined or "collection" in combined else 0
        elif "pack" in combined or "collection" in combined or "130+" in combined:
            media_type = "Collection / Resource Pack"
            is_collection = 1
        elif "manifest" in combined or "specification" in combined or "architecture" in combined or file_type == "Markdown":
            media_type = "Project Documentation / Manifest"
        elif "bom" in combined or ".csv" in combined or "datasheet" in combined:
            media_type = "Hardware Specification / BOM"
        elif "statement" in combined or "bill" in combined or "1040" in combined or "invoice" in combined:
            media_type = "Financial Record / Document"
        elif file_type == "Directory / Folder":
            media_type = "Directory / Resource Pack"
            is_collection = 1
        else:
            media_type = "Book / Ebook"

        return media_type, is_collection

    @staticmethod
    def synthesize_summary(
        clean_title: str,
        subtitle: Optional[str],
        author: Optional[str],
        publisher: Optional[str],
        domain: str,
        subject: str,
        subfield: str,
        genre: str,
        media_type: str,
        edition: str,
        published: Optional[str] = None,
    ) -> str:
        """Generates a cohesive, context-rich abstract for ML embeddings and RAG pipelines."""
        author_str = f" by {author}" if author else ""
        pub_str = f", released in {published}" if published else ""
        publisher_str = f" from {publisher}" if publisher else ""
        sub_desc = f" ({subtitle})" if subtitle else ""

        if "Collection" in media_type or "Pack" in media_type:
            return (
                f"Comprehensive resource bundle and reference collection focusing on {clean_title}{sub_desc}{publisher_str}{pub_str}. "
                f"Covers key principles, curated documentation, and multi-part assets across {subfield} within {domain}."
            )
        if "Course" in media_type:
            return (
                f"Structured educational course and instructional video curriculum covering {clean_title}{sub_desc}{author_str}{publisher_str}. "
                f"Designed to build practical mastery and hands-on proficiency in {subfield}."
            )
        if "Documentation" in media_type:
            return (
                f"System architecture blueprint and diagnostic specification for {clean_title}{sub_desc}. "
                f"Details functional protocols, instrumentation pipelines, and execution frameworks in {domain}."
            )
        if "Financial" in media_type:
            return (
                f"Official fiscal documentation and financial statement record regarding {clean_title}{pub_str}."
            )

        return (
            f"A comprehensive {genre.lower()} on {clean_title}{sub_desc} ({edition}){author_str}{publisher_str}{pub_str}. "
            f"Explores foundational principles, technical methodologies, and practical applications in {subject} and {subfield}."
        )


class OnlineEnricher:
    """Queries Open Library, Crossref, and Google Books APIs with rate limiting and caching."""

    def __init__(
        self,
        cache: MetadataCache,
        delay: float = 0.01,
        timeout: float = 3.5,
        offline_only: bool = False,
        scan_front_matter: bool = True,
    ):
        self.cache = cache
        self.delay = delay
        self.timeout = timeout
        self.offline_only = offline_only
        self.scan_front_matter = scan_front_matter

    def _prepare_search_query(self, title: str) -> Optional[str]:
        q = title.strip()
        # Skip pure dates, numbers, statements, manifests, bills, hash strings
        if re.match(r"^[\d\s\-_.]+$", q):
            return None
        if re.search(r"\b(statement|invoice|bill|1040|tax|manifest|diagnostic\s+ecosystem|bom|two\s+kings\s+manual|202403|202407|12251)\b", q, re.I):
            return None
        if re.match(r"^[0-9a-f]{16,}", q, re.I):
            return None

        q = re.sub(r"^Udemy\s*-\s*", "", q, flags=re.I)
        q = re.sub(r"\[[^\]]+\]", "", q)
        q = re.sub(r"\b(\d+)(?:st|nd|rd|th)\s+(?:edition|ed)\b", "", q, flags=re.I)
        q = re.sub(r"\b(?:pack|collection)[-_\s]*\d+\b", "", q, flags=re.I)
        q = re.sub(r"\b(19\d\d|20\d\d)\b", "", q)
        q = re.sub(r"[^\w\s]", " ", q)
        q = re.sub(r"\s+", " ", q).strip()

        words = [w for w in q.split() if len(w) > 1 and not w.isdigit()]
        if len(words) < 2:
            return None
        return " ".join(words[:6])

    def query_open_library(self, title: str) -> Optional[Dict[str, Any]]:
        search_query = self._prepare_search_query(title)
        if not search_query:
            return None

        url = (
            f"https://openlibrary.org/search.json?"
            f"q={urllib.parse.quote(search_query)}&"
            f"fields=title,author_name,first_publish_year,subject,first_sentence,edition_count,publisher,language&"
            f"limit=2"
        )
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                docs = data.get("docs", [])
                for doc in docs:
                    doc_title = doc.get("title", "")
                    similarity = calculate_title_similarity(search_query, doc_title)
                    if similarity >= 0.35 or search_query.lower() in doc_title.lower() or doc_title.lower() in search_query.lower():
                        authors = doc.get("author_name", [])
                        author_str = ", ".join(authors[:3]) if authors else None
                        pub_year = str(doc.get("first_publish_year")) if doc.get("first_publish_year") else None
                        publishers = doc.get("publisher", [])
                        publisher_str = publishers[0] if publishers else None
                        languages = doc.get("language", [])
                        lang_str = languages[0] if languages else "en"
                        first_sentence = doc.get("first_sentence")
                        summary = first_sentence.get("value") if isinstance(first_sentence, dict) else first_sentence
                        ed_count = doc.get("edition_count")

                        return {
                            "author": author_str,
                            "published": pub_year,
                            "publisher": publisher_str,
                            "language": lang_str,
                            "summary": summary,
                            "revision": ed_count if ed_count and ed_count > 1 else None,
                            "source": "OpenLibrary",
                        }
        except Exception:
            pass
        return None

    def query_crossref(self, title: str) -> Optional[Dict[str, Any]]:
        search_query = self._prepare_search_query(title)
        if not search_query:
            return None

        url = f"https://api.crossref.org/works?query.bibliographic={urllib.parse.quote(search_query)}&rows=1"
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req, context=SSL_CONTEXT, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                items = data.get("message", {}).get("items", [])
                if items:
                    item = items[0]
                    item_titles = item.get("title", [])
                    item_title = item_titles[0] if item_titles else ""
                    if calculate_title_similarity(search_query, item_title) >= 0.4:
                        authors = item.get("author", [])
                        author_names = []
                        for a in authors[:3]:
                            name = f"{a.get('given', '')} {a.get('family', '')}".strip()
                            if name:
                                author_names.append(name)
                        author_str = ", ".join(author_names) if author_names else None
                        published_date = None
                        pub_parts = item.get("published", {}).get("date-parts", [[]])[0]
                        if pub_parts:
                            published_date = str(pub_parts[0])

                        publisher_str = item.get("publisher")

                        return {
                            "author": author_str,
                            "published": published_date,
                            "publisher": publisher_str,
                            "source": "Crossref",
                        }
        except Exception:
            pass
        return None

    def enrich(self, clean_title: str, raw_title: str, source_path: Optional[str] = None, in_dir: int = 0) -> EnrichedBookRecord:
        from compare_books import clean_title as sanitize_title

        # Step 1: Locate disk file and extract front matter if present
        fm: Optional[FrontMatterResult] = None
        target_path: Optional[Path] = None
        if source_path and Path(source_path).is_file():
            target_path = Path(source_path)
        elif raw_title:
            for cand_dir in [DEFAULT_DIRECTORY, Path(".")]:
                p_cand = cand_dir / raw_title
                if p_cand.is_file():
                    target_path = p_cand
                    break

        if self.scan_front_matter and target_path:
            fm = extract_front_matter(target_path)
            if not source_path:
                source_path = str(target_path.resolve())

        # Step 2: ISBN Lookup if ISBN discovered
        isbn_data: Optional[Dict[str, Any]] = None
        if fm and fm.isbn and not self.offline_only:
            isbn_cache_key = f"isbn:{fm.isbn}"
            cached_isbn = self.cache.get(isbn_cache_key)
            if cached_isbn is not None:
                isbn_data = cached_isbn if cached_isbn else None
            else:
                isbn_data = query_online_by_isbn(fm.isbn, timeout=self.timeout)
                self.cache.set(isbn_cache_key, isbn_data if isbn_data else {})

        # Step 3: Resolve title from front matter or ISBN if current title is uninformative
        is_uninf = is_uninformative_title(clean_title)
        candidate_title = None
        if isbn_data and isbn_data.get("title") and is_uninf:
            candidate_title = isbn_data["title"]
        elif fm and fm.has_useful_title and is_uninf:
            candidate_title = fm.title

        if candidate_title:
            clean_title = sanitize_title(candidate_title)

        # Step 4: Local structural extraction
        file_type, file_ext = LocalRuleExtractor.extract_file_info(raw_title, source_path)
        main_title, subtitle = LocalRuleExtractor.extract_subtitle(clean_title)
        if not subtitle and isbn_data and isbn_data.get("subtitle"):
            subtitle = isbn_data["subtitle"]
        elif not subtitle and fm and fm.subtitle:
            subtitle = fm.subtitle

        local_author = (
            (isbn_data.get("author") if isbn_data else None)
            or (fm.author if fm and fm.author else None)
            or LocalRuleExtractor.extract_author(raw_title, clean_title)
        )
        local_publisher = (
            (isbn_data.get("publisher") if isbn_data else None)
            or (fm.publisher if fm and fm.publisher else None)
            or LocalRuleExtractor.extract_publisher(raw_title, clean_title)
        )
        local_published = (
            (isbn_data.get("published") if isbn_data else None)
            or (fm.published if fm and fm.published else None)
            or LocalRuleExtractor.extract_published_year(raw_title, clean_title)
        )
        revision_num, edition_str = LocalRuleExtractor.extract_edition_info(raw_title, clean_title)
        if fm and fm.revision and fm.revision > revision_num:
            revision_num = fm.revision
            edition_str = fm.edition or f"{revision_num}th Edition"

        domain, subject, subfield, genre, audience, concepts, topics = LocalRuleExtractor.match_taxonomy(raw_title, clean_title)
        media_type, is_collection = LocalRuleExtractor.determine_media_classification(raw_title, clean_title, file_type)

        # Step 5: Online API query (if online enabled and not already resolved by ISBN)
        api_data: Optional[Dict[str, Any]] = isbn_data
        cache_key = clean_title.lower().strip()

        if not api_data and not self.offline_only and not is_uninformative_title(clean_title):
            cached = self.cache.get(cache_key)
            if cached is not None:
                # Can be a valid dict or empty dict {} (cached negative lookup)
                api_data = cached if cached else None
            else:
                api_data = self.query_open_library(clean_title)
                if not api_data:
                    api_data = self.query_crossref(clean_title)

                # Store result or empty dict for negative cache
                self.cache.set(cache_key, api_data if api_data else {})
                if self.delay > 0:
                    time.sleep(self.delay)

        # Step 6: Synthesis & Fusion
        author = (api_data.get("author") if api_data and api_data.get("author") else None) or local_author
        publisher = (api_data.get("publisher") if api_data and api_data.get("publisher") else None) or local_publisher
        published = (api_data.get("published") if api_data and api_data.get("published") else None) or local_published
        language = (
            (api_data.get("language") if api_data and api_data.get("language") else None)
            or (fm.language if fm and fm.language else "en")
        )
        if api_data and api_data.get("revision") and api_data["revision"] > revision_num:
            revision_num = api_data["revision"]
            edition_str = f"{revision_num}th Edition"

        # Summary resolution
        api_summary = (api_data.get("summary") if api_data else None) or (fm.summary if fm and fm.summary else None)
        if api_summary and len(clean_html_text(api_summary)) >= 25:
            summary = clean_html_text(api_summary)
        else:
            summary = LocalRuleExtractor.synthesize_summary(
                clean_title=clean_title,
                subtitle=subtitle,
                author=author,
                publisher=publisher,
                domain=domain,
                subject=subject,
                subfield=subfield,
                genre=genre,
                media_type=media_type,
                edition=edition_str,
                published=published,
            )

        source = (
            (api_data.get("source") if api_data else None)
            or (fm.source if fm and fm.source != "unsupported_format" else None)
            or "local+rules"
        )

        return EnrichedBookRecord(
            clean_title=clean_title,
            subtitle=subtitle,
            raw_title=raw_title,
            author=author,
            publisher=publisher,
            published=published,
            edition=edition_str,
            revision=revision_num,
            language=language,
            domain=domain,
            subject=subject,
            subfield=subfield,
            genre=genre,
            topics=topics,
            key_concepts=concepts,
            target_audience=audience,
            media_type=media_type,
            is_collection=is_collection,
            summary=summary,
            file_type=file_type,
            file_extension=file_ext,
            source_path=source_path,
            in_directory=in_dir,
            source=source,
        )


SCHEMA_COLUMNS: List[Tuple[str, str]] = [
    ("id", "INTEGER PRIMARY KEY AUTOINCREMENT"),
    ("clean_title", "TEXT NOT NULL UNIQUE"),
    ("subtitle", "TEXT"),
    ("raw_title", "TEXT NOT NULL"),
    ("author", "TEXT"),
    ("publisher", "TEXT"),
    ("published", "DATE"),
    ("edition", "TEXT"),
    ("revision", "INTEGER DEFAULT 1"),
    ("language", "TEXT DEFAULT 'en'"),
    ("domain", "TEXT"),
    ("subject", "TEXT"),
    ("subfield", "TEXT"),
    ("genre", "TEXT"),
    ("topics", "TEXT"),
    ("key_concepts", "TEXT"),
    ("target_audience", "TEXT"),
    ("media_type", "TEXT"),
    ("is_collection", "INTEGER DEFAULT 0"),
    ("summary", "TEXT"),
    ("file_type", "TEXT"),
    ("file_extension", "TEXT"),
    ("source_path", "TEXT"),
    ("in_directory", "INTEGER DEFAULT 0"),
    ("created_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
    ("updated_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"),
]


def ensure_table_schema(conn: sqlite3.Connection, table_name: str = "books") -> None:
    """Ensures that all ML and media library columns exist in the database."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_cols = {row[1] for row in cursor.fetchall()}

    if not existing_cols:
        col_defs = ", ".join([f"{name} {dtype}" for name, dtype in SCHEMA_COLUMNS])
        cursor.execute(f"CREATE TABLE {table_name} ({col_defs})")
        cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_clean_title ON {table_name}(clean_title);")
        return

    for col_name, col_type in SCHEMA_COLUMNS:
        if col_name == "id":
            continue
        if col_name not in existing_cols:
            base_type = col_type.split()[0]
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {base_type}")

    cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_clean_title ON {table_name}(clean_title);")
    conn.commit()


def enrich_database(
    db_path: Union[str, Path] = DEFAULT_DB_PATH,
    export_csv_path: Optional[Union[str, Path]] = DEFAULT_CSV_PATH,
    offline_only: bool = False,
    scan_front_matter: bool = True,
    delay: float = 0.05,
    max_workers: int = 4,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
) -> Dict[str, Any]:
    """
    Enriches all records in the books SQLite database with rich ML and media metadata.
    """
    import concurrent.futures

    db_file = Path(db_path)
    if not db_file.is_file():
        raise FileNotFoundError(f"Database file not found: {db_file}")

    cache = MetadataCache()
    enricher = OnlineEnricher(
        cache=cache,
        delay=delay,
        offline_only=offline_only,
        scan_front_matter=scan_front_matter,
    )

    with sqlite3.connect(db_file) as conn:
        ensure_table_schema(conn, table_name="books")
        cursor = conn.cursor()

        cursor.execute("SELECT id, clean_title, raw_title, source_path, in_directory FROM books ORDER BY id ASC")
        rows = cursor.fetchall()
        total_rows = len(rows)

        online_count = 0
        local_count = 0
        enriched_results: List[Tuple[int, EnrichedBookRecord]] = []

        if offline_only or max_workers <= 1:
            for idx, (book_id, clean_title, raw_title, source_path, in_dir) in enumerate(rows, 1):
                rec = enricher.enrich(clean_title, raw_title, source_path, in_dir or 0)
                enriched_results.append((book_id, rec))
                if idx % 50 == 0 or idx == total_rows:
                    print(f"[{idx}/{total_rows}] Processed '{clean_title[:40]}' ({rec.source})", flush=True)
        else:
            def process_row(row_item):
                b_id, c_title, r_title, s_path, i_dir = row_item
                return b_id, enricher.enrich(c_title, r_title, s_path, i_dir or 0)

            print(f"Starting parallel enrichment across {total_rows} records with {max_workers} workers...", flush=True)
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(process_row, r): r for r in rows}
                completed = 0
                for f in concurrent.futures.as_completed(futures):
                    b_id, rec = f.result()
                    enriched_results.append((b_id, rec))
                    completed += 1
                    if completed % 50 == 0 or completed == total_rows:
                        print(f"[{completed}/{total_rows}] Enriched '{rec.clean_title[:40]}' -> {rec.domain} | {rec.subject}", flush=True)

        for book_id, record in enriched_results:
            if "OpenLibrary" in record.source or "Crossref" in record.source or "GoogleBooks" in record.source:
                online_count += 1
            else:
                local_count += 1

            cursor.execute(
                """
                UPDATE books
                SET clean_title = ?,
                    subtitle = ?,
                    author = ?,
                    publisher = ?,
                    published = ?,
                    edition = ?,
                    revision = ?,
                    language = ?,
                    domain = ?,
                    subject = ?,
                    subfield = ?,
                    genre = ?,
                    topics = ?,
                    key_concepts = ?,
                    target_audience = ?,
                    media_type = ?,
                    is_collection = ?,
                    summary = ?,
                    file_type = ?,
                    file_extension = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (
                    record.clean_title,
                    record.subtitle,
                    record.author,
                    record.publisher,
                    record.published,
                    record.edition,
                    record.revision,
                    record.language,
                    record.domain,
                    record.subject,
                    record.subfield,
                    record.genre,
                    record.topics,
                    record.key_concepts,
                    record.target_audience,
                    record.media_type,
                    record.is_collection,
                    record.summary,
                    record.file_type,
                    record.file_extension,
                    book_id,
                ),
            )

        conn.commit()
        cache.save()

    # Export updated CSV
    if export_csv_path:
        from compare_books import export_db_to_csv
        export_db_to_csv(db_path=db_path, output_path=export_csv_path)

    return {
        "total_records": total_rows,
        "online_enriched": online_count,
        "local_rule_enriched": local_count,
        "db_path": str(db_file.resolve()),
        "csv_path": str(Path(export_csv_path).resolve()) if export_csv_path else None,
    }


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enrich media library database with ML features, categories, and summaries."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH})",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Path to export enriched CSV (default: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument(
        "--offline-only",
        action="store_true",
        help="Run rule-based extraction offline without network API queries",
    )
    parser.add_argument(
        "--no-scan-front-matter",
        action="store_true",
        help="Disable automatic front matter inspection for PDF/EPUB files on disk",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Rate-limiting delay between API requests in seconds (default: 0.01)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=12,
        help="Number of concurrent worker threads for enrichment (default: 12)",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    parsed = parse_args(args)
    print("=" * 70)
    print("      STARTING MEDIA LIBRARY & ML METADATA ENRICHMENT")
    print("=" * 70)
    print(f"Database Path: {parsed.db_path}")
    print(f"Export CSV:    {parsed.export_csv}")
    print(f"Mode:          {'Offline-only' if parsed.offline_only else 'Online APIs (OpenLibrary/Crossref) + Fallback'}")
    print(f"Workers:       {parsed.workers}")
    print("-" * 70)

    try:
        stats = enrich_database(
            db_path=parsed.db_path,
            export_csv_path=parsed.export_csv,
            offline_only=parsed.offline_only,
            scan_front_matter=not parsed.no_scan_front_matter,
            delay=parsed.delay,
            max_workers=parsed.workers,
        )
        print("=" * 70)
        print("             ENRICHMENT COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Total Records Enriched:     {stats['total_records']}")
        print(f"Online API Matches:         {stats['online_enriched']}")
        print(f"Local & Heuristic Matches:  {stats['local_rule_enriched']}")
        print(f"Database Updated:           {stats['db_path']}")
        if stats['csv_path']:
            print(f"CSV Exported:               {stats['csv_path']}")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"Error during enrichment: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
