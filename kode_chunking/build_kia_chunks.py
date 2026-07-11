import json
import re
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
import statistics

# ============================================================================
# CONFIGURATION
# ============================================================================
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from config import CHUNK_OVERLAP, CHUNK_SIZE, SCENARIO_NAME

SOURCE_PDF = str(BASE_DIR / "sumber_data" / "Buku_KIA_2024 (1).pdf")
OUT_FILE = str(BASE_DIR / "hasil_chunking" / "kia_chunks.json")
SOURCE_DIR = BASE_DIR / "sumber_data"
OUTPUT_DIR = BASE_DIR / "hasil_chunking"

CHUNK_PROFILES = {
    "strict": {"min_words": 200, "max_words": 500, "min_output_words": 120, "merge_under": 200, "merge_max_total": 500},
    "balanced": {"min_words": 120, "max_words": 350, "min_output_words": 100, "merge_under": 120, "merge_max_total": 380},
    "high_recall": {"min_words": 90, "max_words": 280, "min_output_words": 80, "merge_under": 90, "merge_max_total": 320},
}

SECTION_ANCHORS = [
    "tentang buku kia", "1000 hari pertama kehidupan", "trimester 1", "trimester 2", "trimester 3",
    "yang harus dilakukan", "yang akan dialami", "mengapa harus dilakukan", "tanda bahaya",
    "tidak boleh dilakukan", "persiapan melahirkan", "tanda awal melahirkan", "proses melahirkan",
    "masa nifas", "menyusui", "asi eksklusif", "bayi baru lahir", "0 - 6 bulan", "6 - 12 bulan",
    "12 - 24 bulan", "2 - 6 tahun", "pantau tumbuh kembang", "stimulasi", "imunisasi", "keluarga berencana",
]

HEADING_BLACKLIST = {
    "daftar isi", "kata pengantar", "pendahuluan", "kata sambutan", "daftar pustaka",
    "referensi", "bibliografi", "lampiran", "glosarium", "indeks", "copyright", "hak cipta",
}

STOPWORDS = {
    "yang", "dan", "atau", "adalah", "untuk", "dengan", "pada", "dari", "ke", "di", "ini", "itu",
    "dalam", "agar", "jika", "serta", "oleh", "lebih", "setelah", "saat", "masa", "sudah", "akan", "juga",
    "seperti", "karena", "bisa", "dapat", "hal", "bagi", "kali", "hari", "tahun", "bulan", "ibu", "anak",
    "tidak", "boleh", "harus", "segera", "halaman",
}

TITLE_MAP = {
    "kehamilan_umum": "Kehamilan - Umum",
    "kehamilan_trimester_1": "Kehamilan - Trimester 1",
    "kehamilan_trimester_2": "Kehamilan - Trimester 2",
    "kehamilan_trimester_3": "Kehamilan - Trimester 3",
    "kehamilan_nutrisi": "Kehamilan - Nutrisi",
    "kehamilan_pemeriksaan": "Kehamilan - Pemeriksaan",
    "kehamilan_aktivitas": "Kehamilan - Aktivitas",
    "kehamilan_tanda_bahaya": "Kehamilan - Tanda Bahaya",
    "persalinan_persiapan": "Persalinan - Persiapan",
    "persalinan_tanda_awal": "Persalinan - Tanda Awal",
    "persalinan_proses": "Persalinan - Proses",
    "persalinan_tanda_bahaya": "Persalinan - Tanda Bahaya",
    "nifas_perawatan": "Nifas - Perawatan",
    "nifas_pemulihan": "Nifas - Pemulihan",
    "nifas_nutrisi": "Nifas - Nutrisi",
    "nifas_tanda_bahaya": "Nifas - Tanda Bahaya",
    "mental_ibu_hamil": "Kesehatan Mental - Ibu Hamil",
    "mental_pasca_melahirkan": "Kesehatan Mental - Pasca Melahirkan",
    "mental_baby_blues": "Kesehatan Mental - Baby Blues",
    "mental_depresi": "Kesehatan Mental - Depresi",
    "menyusui_pengertian": "Menyusui - Pengertian",
    "menyusui_manfaat": "Menyusui - Manfaat",
    "menyusui_teknik": "Menyusui - Teknik",
    "menyusui_masalah": "Menyusui - Masalah",
    "menyusui_penyimpanan_asi": "Menyusui - Penyimpanan ASI",
    "bayi_baru_lahir_perawatan": "Bayi Baru Lahir - Perawatan",
    "bayi_baru_lahir_tanda_bahaya": "Bayi Baru Lahir - Tanda Bahaya",
    "bayi_baru_lahir_pemeriksaan": "Bayi Baru Lahir - Pemeriksaan",
    "bayi_asi_eksklusif": "Bayi 0-6 Bulan - ASI Eksklusif",
    "bayi_perkembangan": "Bayi 0-6 Bulan - Perkembangan",
    "bayi_perawatan": "Bayi 0-6 Bulan - Perawatan",
    "bayi_tanda_bahaya": "Bayi 0-6 Bulan - Tanda Bahaya",
    "balita_perkembangan": "Balita - Perkembangan",
    "balita_nutrisi": "Balita - Nutrisi",
    "balita_stimulasi": "Balita - Stimulasi",
    "imunisasi_pengertian": "Imunisasi - Pengertian",
    "imunisasi_jadwal": "Imunisasi - Jadwal",
    "imunisasi_manfaat": "Imunisasi - Manfaat",
    "kb_pengertian": "Keluarga Berencana - Pengertian",
    "kb_tujuan": "Keluarga Berencana - Tujuan",
    "kb_jenis": "Keluarga Berencana - Jenis",
    "kb_manfaat": "Keluarga Berencana - Manfaat",
    "tumbuh_kembang_fisik": "Tumbuh Kembang - Fisik",
    "tumbuh_kembang_motorik": "Tumbuh Kembang - Motorik",
    "tumbuh_kembang_kognitif": "Tumbuh Kembang - Kognitif",
}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def clean_line(line: str) -> str:
    line = line.replace("✔", "").replace("✓", "")
    line = line.replace("•", "- ").replace("●", "- ")
    return re.sub(r"\s+", " ", line).strip()

def extract_lines(pdf_path: str):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Failed to open PDF: {e}")
        return []
    pages = []
    for pno in range(doc.page_count):
        page = doc[pno]
        d = page.get_text("dict")
        page_spans, lines = [], []
        for block in d.get("blocks", []):
            for l in block.get("lines", []):
                parts, sizes = [], []
                for span in l.get("spans", []):
                    txt = span.get("text", "")
                    if txt.strip():
                        parts.append(txt)
                        sizes.append(span.get("size", 0))
                        page_spans.append(span.get("size", 0))
                if parts:
                    lines.append({"text": clean_line("".join(parts)), "size": max(sizes) if sizes else 0})
        valid = [s for s in page_spans if 7 <= s <= 25]
        median = statistics.median(valid) if valid else 0
        pages.append((pno + 1, lines, median))
    return pages

def line_is_heading(line: str, meta: dict, page_median: float) -> bool:
    s = line.strip()
    if len(s) < 4 or len(s) > 160:
        return False
    low = s.lower()
    if any(blk in low for blk in HEADING_BLACKLIST):
        return False
    if any(anchor in low for anchor in SECTION_ANCHORS):
        return True
    size = meta.get("size", 0)
    if page_median and size >= page_median * 1.15:
        return True
    if re.match(r"^\s*(\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])[\s\.\)\-]", s):
        return True
    return False

def classify_content_type(title: str, content: str) -> str:
    t = f"{title} {content}".lower()
    if any(k in t for k in ("kata pengantar", "kata sambutan", "pendahuluan", "daftar isi")):
        return "frontmatter"
    if any(k in t for k in ("daftar pustaka", "referensi", "bibliografi")):
        return "backmatter"
    if "lampiran" in t or "appendix" in t:
        return "appendix"
    if "tanda bahaya" in t or "darurat" in t:
        return "safety"
    return "edukasi"

def split_sections(pages_lines):
    sections, cur_lines, cur_title, cur_start = [], [], "Document", None
    def flush():
        nonlocal cur_lines, cur_title, cur_start
        content = "\n".join(cur_lines).strip()
        ctype = classify_content_type(cur_title, content)
        if ctype not in ("frontmatter", "backmatter") and len(content.split()) >= 40:
            sections.append((cur_title, content, cur_start, ctype))
        cur_lines, cur_title, cur_start = [], "Document", None

    for pno, lines, median in pages_lines:
        if cur_start is None: cur_start = pno
        cur_lines.append(f"[HALAMAN {pno}]")
        for meta in lines:
            txt = meta["text"]
            if line_is_heading(txt, meta, median) and len(" ".join(cur_lines).split()) >= 60:
                flush()
                cur_title, cur_start = txt, pno
            else:
                cur_lines.append(txt)
    flush()
    return sections

def split_special_blocks(content: str) -> List[str]:
    low = content.lower()
    if "tanda bahaya" in low or "yang harus dilakukan" in low or "tidak boleh" in low:
        parts = re.split(r"(?i)(?=tanda bahaya|yang harus dilakukan|hal-hal yang tidak boleh|tidak boleh dilakukan)", content)
        return [p.strip() for p in parts if len(p.split()) >= 40]
    return [content]


def normalize_spaces(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()

def best_split_point(text: str, limit: int) -> int:
    if len(text) <= limit:
        return len(text)

    search_start = max(0, limit - max(80, limit // 4))
    window = text[search_start:limit]
    candidates = []

    for token in ["\n\n", "\n"]:
        pos = window.rfind(token)
        if pos != -1:
            candidates.append(search_start + pos + len(token))

    sentence_hits = list(re.finditer(r"(?<=[.!?])\s+", window))
    if sentence_hits:
        candidates.append(search_start + sentence_hits[-1].end())

    space_pos = window.rfind(" ")
    if space_pos != -1:
        candidates.append(search_start + space_pos + 1)

    usable = [c for c in candidates if 0 < c < limit]
    return max(usable) if usable else limit


def chunk_text(text: str, max_chars: int = CHUNK_SIZE, overlap_chars: int = CHUNK_OVERLAP) -> List[str]:
    text = normalize_spaces(text)
    if not text:
        return []

    max_chars = max(1, max_chars)
    overlap_chars = max(0, min(overlap_chars, max_chars - 1))

    chunks = []
    start = 0
    total_len = len(text)

    while start < total_len:
        end = min(start + max_chars, total_len)
        if end < total_len:
            split_at = best_split_point(text[start:end], end - start)
            end = start + split_at

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= total_len:
            break

        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks

def classify_category_sub(title: str, content: str) -> Tuple[str, str]:
    t = f"{title} {content}".lower()
    if "trimester 1" in t or "1-3 bulan" in t: return "kehamilan", "kehamilan_trimester_1"
    if "trimester 2" in t or "4-6 bulan" in t: return "kehamilan", "kehamilan_trimester_2"
    if "trimester 3" in t or "7-9 bulan" in t: return "kehamilan", "kehamilan_trimester_3"
    if "tanda bahaya" in t and any(k in t for k in ["hamil", "kehamilan", "trimester"]): return "kehamilan", "kehamilan_tanda_bahaya"
    if "pemeriksaan" in t or "usg" in t or "anc" in t: return "kehamilan", "kehamilan_pemeriksaan"
    if ("gizi" in t or "makan" in t or "ttd" in t) and "hamil" in t: return "kehamilan", "kehamilan_nutrisi"
    if "aktivitas" in t or "istirahat" in t or "olahraga" in t: return "kehamilan", "kehamilan_aktivitas"
    if "kehamilan" in t or "1000 hari pertama" in t: return "kehamilan", "kehamilan_umum"

    if "tanda awal melahirkan" in t: return "persalinan", "persalinan_tanda_awal"
    if "persiapan melahirkan" in t or "rencana persalinan" in t: return "persalinan", "persalinan_persiapan"
    if "tanda bahaya" in t and "melahir" in t: return "persalinan", "persalinan_tanda_bahaya"
    if "proses melahirkan" in t or "persalinan" in t: return "persalinan", "persalinan_proses"

    if "nifas" in t and "tanda bahaya" in t: return "nifas", "nifas_tanda_bahaya"
    if "nifas" in t and any(k in t for k in ["pemulihan", "perubahan tubuh"]): return "nifas", "nifas_pemulihan"
    if "nifas" in t and any(k in t for k in ["makan", "gizi", "nutrisi"]): return "nifas", "nifas_nutrisi"
    if "nifas" in t: return "nifas", "nifas_perawatan"

    if "baby blues" in t: return "kesehatan_mental", "mental_baby_blues"
    if "depresi" in t: return "kesehatan_mental", "mental_depresi"
    if "pasca melahirkan" in t or "postpartum" in t: return "kesehatan_mental", "mental_pasca_melahirkan"
    if "cemas" in t or "stres" in t or "mental" in t: return "kesehatan_mental", "mental_ibu_hamil"

    if "bayi baru lahir" in t and "tanda bahaya" in t: return "bayi_baru_lahir", "bayi_baru_lahir_tanda_bahaya"
    if "bayi baru lahir" in t and any(k in t for k in ["pemeriksaan", "kunjungan"]): return "bayi_baru_lahir", "bayi_baru_lahir_pemeriksaan"
    if "bayi baru lahir" in t: return "bayi_baru_lahir", "bayi_baru_lahir_perawatan"
    if "0 - 6 bulan" in t and "tanda bahaya" in t: return "bayi_0_6_bulan", "bayi_tanda_bahaya"
    if "0 - 6 bulan" in t and "perkembangan" in t: return "bayi_0_6_bulan", "bayi_perkembangan"
    if "0 - 6 bulan" in t and "asi" in t: return "bayi_0_6_bulan", "bayi_asi_eksklusif"
    if "0 - 6 bulan" in t: return "bayi_0_6_bulan", "bayi_perawatan"

    if "imunisasi" in t and "jadwal" in t: return "imunisasi", "imunisasi_jadwal"
    if "imunisasi" in t and "manfaat" in t: return "imunisasi", "imunisasi_manfaat"
    if "imunisasi" in t: return "imunisasi", "imunisasi_pengertian"
    if "keluarga berencana" in t and "jenis" in t: return "keluarga_berencana", "kb_jenis"
    if "keluarga berencana" in t and "manfaat" in t: return "keluarga_berencana", "kb_manfaat"
    if "keluarga berencana" in t and "tujuan" in t: return "keluarga_berencana", "kb_tujuan"
    if "keluarga berencana" in t or "kontrasepsi" in t or re.search(r"\bkb\b", t): return "keluarga_berencana", "kb_pengertian"

    if "balita" in t and "stimulasi" in t: return "balita", "balita_stimulasi"
    if "balita" in t and any(k in t for k in ["nutrisi", "gizi", "makan"]): return "balita", "balita_nutrisi"
    if "balita" in t: return "balita", "balita_perkembangan"
    if "tumbuh kembang" in t and any(k in t for k in ["bahasa", "kognitif"]): return "tumbuh_kembang", "tumbuh_kembang_kognitif"
    if "tumbuh kembang" in t and "motorik" in t: return "tumbuh_kembang", "tumbuh_kembang_motorik"
    if "tumbuh kembang" in t or any(k in t for k in ["berat badan", "tinggi badan", "lingkar kepala"]): return "tumbuh_kembang", "tumbuh_kembang_fisik"

    if "penyimpanan asi" in t or "asi perah" in t: return "menyusui", "menyusui_penyimpanan_asi"
    if "teknik menyusui" in t or "perlekatan" in t or ("posisi" in t and "menyusui" in t): return "menyusui", "menyusui_teknik"
    if "masalah menyusui" in t or "mastitis" in t or "puting lecet" in t: return "menyusui", "menyusui_masalah"
    if "manfaat asi" in t or "manfaat menyusui" in t: return "menyusui", "menyusui_manfaat"
    if "menyusui" in t or "air susu ibu" in t or "asi eksklusif" in t: return "menyusui", "menyusui_pengertian"

    return "kehamilan", "kehamilan_umum"

def classify_type_priority(title: str, content: str) -> Tuple[str, str]:
    t = f"{title} {content}".lower()
    if "tanda bahaya" in t: return "tanda_bahaya", "kritis"
    if any(k in t for k in ["tidak boleh", "hindari", "jangan", "dilarang"]): return "larangan", "penting"
    if any(k in t for k in ["yang harus dilakukan", "anjuran", "lakukan", "segera periksa", "pastikan", "minum"]): return "anjuran", "penting"
    if any(k in t for k in ["perkembangan", "tumbuh kembang", "yang akan dialami", "penanda perkembangan"]): return "perkembangan", "umum"
    return "edukasi", "umum"

def extract_keywords(title: str, content: str, max_kw: int = 8) -> List[str]:
    text = re.sub(r"[^a-z0-9\s]", " ", (title + " " + content).lower())
    freq = {}
    for w in text.split():
        if len(w) > 2 and w not in STOPWORDS and not w.isdigit():
            freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:max_kw]]

def normalize_chunks(records: List[Dict], min_chars: int = None, max_chars: int = CHUNK_SIZE) -> List[Dict]:
    if min_chars is None:
        min_chars = max(120, max_chars // 2)

    merged = []
    for rec in records:
        cur_len = len(rec["content"])
        if merged and cur_len < min_chars and merged[-1]["category"] == rec["category"]:
            prev_len = len(merged[-1]["content"])
            if prev_len + cur_len + 2 <= max_chars:
                merged[-1]["content"] += "\n\n" + rec["content"]
                merged[-1]["keywords"] = list(dict.fromkeys(merged[-1]["keywords"] + rec["keywords"]))[:10]
                if rec["priority"] == "kritis":
                    merged[-1]["priority"] = "kritis"
                    merged[-1]["type"] = "tanda_bahaya"
                continue
        merged.append(rec)

    final = []
    for rec in merged:
        if len(rec["content"]) <= max_chars:
            final.append(rec)
            continue

        pieces = chunk_text(rec["content"], max_chars=max_chars, overlap_chars=0)
        if not pieces:
            final.append(rec)
            continue

        for idx, piece in enumerate(pieces, start=1):
            nr = dict(rec)
            nr["title"] = f"{rec['title']} - Bagian {idx}" if idx > 1 else rec["title"]
            nr["content"] = piece
            final.append(nr)
    return final


def slugify_filename(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower())
    return slug.strip("_") or "chunk"


def build_chunks(source_path: str, output_path: str, profile: str) -> bool:
    if not Path(source_path).exists():
        logger.error(f"Source PDF not found: {source_path}")
        return False

    logger.info("Extracting text and layout metadata...")
    pages_lines = extract_lines(source_path)
    if not pages_lines:
        return False

    logger.info("Splitting into sections...")
    sections = split_sections(pages_lines)

    logger.info("Chunking and classifying...")
    records = []
    for sec_title, sec_content, sec_page, sec_type in sections:
        for idx, part in enumerate(split_special_blocks(sec_content), 1):
            chunks = chunk_text(part, max_chars=CHUNK_SIZE, overlap_chars=CHUNK_OVERLAP)
            for i, ch in enumerate(chunks, 1):
                title = sec_title
                if len(chunks) > 1:
                    title = f"{sec_title} - Bagian {i}"
                if idx > 1:
                    title = f"{title} (Subtopik {idx})"

                cat, sub = classify_category_sub(title, ch)
                typ, pri = classify_type_priority(title, ch)
                rec = {
                    "title": TITLE_MAP.get(sub, title),
                    "content": ch,
                    "category": cat,
                    "sub_category": sub,
                    "type": typ,
                    "content_type": sec_type,
                    "start_page": sec_page,
                    "keywords": extract_keywords(title, ch),
                    "priority": pri,
                    "source": source_path,
                }
                records.append(rec)

    logger.info("Normalizing chunk sizes (merge/split)...")
    final = normalize_chunks(records, max_chars=CHUNK_SIZE)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    lengths = [len(r["content"]) for r in final]
    if lengths:
        logger.info(f"Done | Sections: {len(sections)} | Chunks: {len(final)} | Chars: {min(lengths)}/{max(lengths)}")
    else:
        logger.info(f"Done | Sections: {len(sections)} | Chunks: 0")
    return True

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build initial chunks from a maternal-child health PDF")
    parser.add_argument("--source", default=SOURCE_PDF, help="Source PDF file path")
    parser.add_argument("--out", default=OUT_FILE, help="Output JSON file path")
    parser.add_argument("--source-dir", help="Process every PDF in this folder")
    parser.add_argument("--out-dir", help="Output folder for batch mode")
    parser.add_argument("--profile", default="balanced", choices=tuple(CHUNK_PROFILES.keys()), help="Chunking profile")
    args = parser.parse_args()

    logger.info(
        f"Using config scenario {SCENARIO_NAME} | chunk_size={CHUNK_SIZE} chars | overlap={CHUNK_OVERLAP} chars"
    )

    if args.source_dir:
        source_dir = Path(args.source_dir)
        output_dir = Path(args.out_dir) if args.out_dir else OUTPUT_DIR
        pdf_files = sorted(p for p in source_dir.iterdir() if p.is_file() and p.suffix.lower() == ".pdf")

        if not pdf_files:
            logger.error(f"No PDF files found in: {source_dir}")
            return

        for pdf_path in pdf_files:
            out_name = f"{slugify_filename(pdf_path.stem)}_chunks_{args.profile}.json"
            out_path = output_dir / out_name
            logger.info(f"Processing batch file: {pdf_path.name}")
            build_chunks(str(pdf_path), str(out_path), args.profile)
        return

    build_chunks(args.source, args.out, args.profile)

if __name__ == "__main__":
    main()