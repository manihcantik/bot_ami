import json
import re
import argparse
from pathlib import Path

import fitz
import statistics

BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_PDF = str(BASE_DIR / "sumber_data" / "Buku_KIA_2024 (1).pdf")
OUT_FILE = str(BASE_DIR / "hasil_chunking" / "kia_chunks.json")


CHUNK_PROFILES = {
    "strict": {
        "min_words": 200,
        "max_words": 500,
        "min_output_words": 120,
        "merge_under": 200,
        "merge_max_total": 500,
    },
    "balanced": {
        "min_words": 120,
        "max_words": 350,
        "min_output_words": 100,
        "merge_under": 120,
        "merge_max_total": 380,
    },
    "high_recall": {
        "min_words": 90,
        "max_words": 280,
        "min_output_words": 80,
        "merge_under": 90,
        "merge_max_total": 320,
    },
}


SECTION_ANCHORS = [
    "tentang buku kia",
    "1000 hari pertama kehidupan",
    "trimester 1",
    "trimester 2",
    "trimester 3",
    "yang harus dilakukan",
    "yang akan dialami",
    "mengapa harus dilakukan",
    "tanda bahaya",
    "tidak boleh dilakukan",
    "persiapan melahirkan",
    "tanda awal melahirkan",
    "proses melahirkan",
    "masa nifas",
    "menyusui",
    "asi eksklusif",
    "bayi baru lahir",
    "0 - 6 bulan",
    "6 - 12 bulan",
    "12 - 24 bulan",
    "2 - 6 tahun",
    "pantau tumbuh kembang",
    "stimulasi",
    "imunisasi",
    "keluarga berencana",
]


# Headings/sections that indicate front/back matter and should be filtered out
HEADING_BLACKLIST = {
    "daftar isi",
    "kata pengantar",
    "pendahuluan",
    "kata sambutan",
    "daftar pustaka",
    "referensi",
    "bibliografi",
    "lampiran",
    "glosarium",
    "indeks",
    "copyright",
    "hak cipta",
}

def clean_line(line: str) -> str:
    line = line.replace("✔", "").replace("✓", "")
    line = line.replace("•", "- ").replace("●", "- ")
    line = re.sub(r"\s+", " ", line).strip()
    return line


def extract_lines(pdf_path: str):
    """Extract lines along with approximate font-size metadata per line.

    Returns list of tuples: (page_number, lines_list, page_median_font_size)
    where lines_list is list of dicts: {"text": str, "size": float}
    """
    doc = fitz.open(pdf_path)
    pages = []
    for pno in range(doc.page_count):
        page = doc[pno]
        d = page.get_text("dict")
        page_spans = []
        lines = []
        for block in d.get("blocks", []):
            for l in block.get("lines", []):
                parts = []
                sizes = []
                for span in l.get("spans", []):
                    txt = span.get("text", "")
                    if txt.strip():
                        parts.append(txt)
                        sizes.append(span.get("size", 0))
                        page_spans.append(span.get("size", 0))
                if parts:
                    text = clean_line("".join(parts))
                    if text:
                        lines.append({"text": text, "size": max(sizes) if sizes else 0})
        # Filter out extreme font sizes (footnotes or huge decorative headers)
        valid_sizes = [s for s in page_spans if 7 <= s <= 25]
        median_size = statistics.median(valid_sizes) if valid_sizes else 0
        pages.append((pno + 1, lines, median_size))
    return pages


def line_is_heading_universal(line: str, meta: dict, page_median: float) -> bool:
    s = line.strip()
    if len(s) < 4 or len(s) > 160:
        return False
    low = s.lower()
    # Blacklist common front/back matter headings
    for blk in HEADING_BLACKLIST:
        if blk in low:
            return False

    # If matches well-known anchors still consider heading
    if any(anchor in low for anchor in SECTION_ANCHORS):
        return True

    # Font-size based heuristic: significantly larger than page median
    size = meta.get("size", 0)
    if page_median and size and size >= page_median * 1.15:
        return True

    # Numbered headings e.g., 1., 1.2, I., A)
    # note: place '-' at the end of character class to avoid range parsing
    if re.match(r"^\s*(\d+(?:\.\d+)*|[IVXLCDM]+|[A-Z])[\s\.\)\-]", s):
        return True

    # UPPERCASE heavy heuristic
    alpha = [c for c in s if c.isalpha()]
    if alpha:
        upper_ratio = sum(1 for c in alpha if c.isupper()) / len(alpha)
        if upper_ratio >= 0.65 and len(s.split()) <= 10:
            return True

    # Title Case short line (likely a heading)
    words = s.split()
    if 1 < len(words) <= 10:
        titlecase_count = sum(1 for w in words if w[0].isupper())
        if titlecase_count >= max(1, len(words) - 1):
            return True

    return False


def split_sections(pages_lines):
    sections = []
    cur_title = "Document"
    cur_lines = []
    cur_start_page = None

    def flush(title_override=None, start_page=None):
        nonlocal cur_lines, cur_title, cur_start_page
        title = title_override or cur_title
        content = "\n".join(cur_lines).strip()
        # classify front/back matter and skip if detected
        ctype = classify_content_type(title, content)
        if ctype in ("frontmatter", "backmatter"):
            cur_lines = []
            return
        if len(content.split()) >= 40:
            sections.append((title, content, cur_start_page if start_page is None else start_page, ctype))
        cur_lines = []
        cur_title = "Document"
        cur_start_page = None

    for page_num, lines, median in pages_lines:
        if cur_start_page is None:
            cur_start_page = page_num
        cur_lines.append(f"[HALAMAN {page_num}]")
        for ln_meta in lines:
            ln = ln_meta["text"]
            if line_is_heading_universal(ln, ln_meta, median) and len(" ".join(cur_lines).split()) >= 60:
                flush()
                cur_title = ln
                cur_start_page = page_num
            else:
                cur_lines.append(ln)

    flush()
    return sections


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


def split_special_blocks(title: str, content: str):
    low = content.lower()
    # Keep danger / recommendation / restriction separated when co-located.
    if "tanda bahaya" in low and ("yang harus dilakukan" in low or "tidak boleh" in low):
        parts = re.split(r"(?i)(?=tanda bahaya|yang harus dilakukan|hal-hal yang tidak boleh|tidak boleh dilakukan)", content)
        out = [p.strip() for p in parts if len(p.split()) >= 40]
        if out:
            return out
    if "yang harus dilakukan" in low and "tidak boleh" in low:
        parts = re.split(r"(?i)(?=yang harus dilakukan|hal-hal yang tidak boleh|tidak boleh dilakukan)", content)
        out = [p.strip() for p in parts if len(p.split()) >= 40]
        if out:
            return out
    return [content]


def chunk_text(text: str, min_words=200, max_words=500, min_output_words=120):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return []

    chunks = []
    cur = []
    cur_words = 0

    for p in paragraphs:
        words = p.split()
        pw = len(words)

        if pw > max_words:
            if cur:
                chunks.append("\n\n".join(cur))
                cur = []
                cur_words = 0
            # Split long paragraph by sentence boundaries.
            sents = re.split(r"(?<=[.!?])\s+", p)
            tmp = []
            tmp_w = 0
            for s in sents:
                sw = len(s.split())
                if tmp_w + sw > max_words and tmp_w >= min_words:
                    chunks.append(" ".join(tmp).strip())
                    tmp = [s]
                    tmp_w = sw
                else:
                    tmp.append(s)
                    tmp_w += sw
            if tmp:
                if chunks and tmp_w < min_words and len(chunks[-1].split()) + tmp_w <= (max_words + 20):
                    chunks[-1] = chunks[-1].rstrip() + " " + " ".join(tmp).strip()
                else:
                    chunks.append(" ".join(tmp).strip())
            continue

        if cur_words + pw > max_words and cur_words >= min_words:
            chunks.append("\n\n".join(cur))
            cur = [p]
            cur_words = pw
        else:
            cur.append(p)
            cur_words += pw

    if cur:
        if chunks and cur_words < min_words and len(chunks[-1].split()) + cur_words <= (max_words + 20):
            chunks[-1] = chunks[-1].rstrip() + "\n\n" + "\n\n".join(cur)
        else:
            chunks.append("\n\n".join(cur))

    return [c.strip() for c in chunks if len(c.split()) >= min_output_words]


def classify_category_sub(title: str, content: str):
    t = f"{title} {content}".lower()

    if "trimester 1" in t or "1-3 bulan" in t:
        return "kehamilan", "kehamilan_trimester_1"
    if "trimester 2" in t or "4-6 bulan" in t:
        return "kehamilan", "kehamilan_trimester_2"
    if "trimester 3" in t or "7-9 bulan" in t:
        return "kehamilan", "kehamilan_trimester_3"
    if "tanda bahaya" in t and ("hamil" in t or "trimester" in t or "kehamilan" in t):
        return "kehamilan", "kehamilan_tanda_bahaya"
    if "pemeriksaan" in t or "usg" in t or "anc" in t:
        return "kehamilan", "kehamilan_pemeriksaan"
    if "porsi makan" in t or "tablet tambah darah" in t or "gizi" in t and "hamil" in t:
        return "kehamilan", "kehamilan_nutrisi"
    if "aktivitas" in t or "istirahat" in t or "olahraga" in t:
        return "kehamilan", "kehamilan_aktivitas"
    if "kehamilan" in t or "1000 hari pertama" in t:
        return "kehamilan", "kehamilan_umum"

    if "tanda awal melahirkan" in t:
        return "persalinan", "persalinan_tanda_awal"
    if "persiapan melahirkan" in t or "rencana persalinan" in t:
        return "persalinan", "persalinan_persiapan"
    if "tanda bahaya" in t and "melahir" in t:
        return "persalinan", "persalinan_tanda_bahaya"
    if "proses melahirkan" in t or "persalinan" in t:
        return "persalinan", "persalinan_proses"

    if "masa nifas" in t and "tanda bahaya" in t:
        return "nifas", "nifas_tanda_bahaya"
    if "masa nifas" in t and ("gizi" in t or "makan" in t):
        return "nifas", "nifas_nutrisi"
    if "masa nifas" in t and ("pemulihan" in t or "perubahan tubuh" in t):
        return "nifas", "nifas_pemulihan"
    if "masa nifas" in t or "nifas" in t:
        return "nifas", "nifas_perawatan"

    if "baby blues" in t:
        return "kesehatan_mental", "mental_baby_blues"
    if "depresi" in t:
        return "kesehatan_mental", "mental_depresi"
    if "pasca melahirkan" in t or "postpartum" in t:
        return "kesehatan_mental", "mental_pasca_melahirkan"
    if "cemas" in t or "stres" in t or "mental" in t:
        return "kesehatan_mental", "mental_ibu_hamil"

    if "bayi baru lahir" in t and "tanda bahaya" in t:
        return "bayi_baru_lahir", "bayi_baru_lahir_tanda_bahaya"
    if "bayi baru lahir" in t and ("pemeriksaan" in t or "kunjungan" in t):
        return "bayi_baru_lahir", "bayi_baru_lahir_pemeriksaan"
    if "bayi baru lahir" in t:
        return "bayi_baru_lahir", "bayi_baru_lahir_perawatan"

    if "0 - 6 bulan" in t and "asi" in t:
        return "bayi_0_6_bulan", "bayi_asi_eksklusif"
    if "0 - 6 bulan" in t and "tanda bahaya" in t:
        return "bayi_0_6_bulan", "bayi_tanda_bahaya"
    if "0 - 6 bulan" in t and "perkembangan" in t:
        return "bayi_0_6_bulan", "bayi_perkembangan"
    if "0 - 6 bulan" in t:
        return "bayi_0_6_bulan", "bayi_perawatan"

    if "imunisasi" in t and "jadwal" in t:
        return "imunisasi", "imunisasi_jadwal"
    if "imunisasi" in t and "manfaat" in t:
        return "imunisasi", "imunisasi_manfaat"
    if "imunisasi" in t:
        return "imunisasi", "imunisasi_pengertian"

    if "keluarga berencana" in t and "jenis" in t:
        return "keluarga_berencana", "kb_jenis"
    if "keluarga berencana" in t and "manfaat" in t:
        return "keluarga_berencana", "kb_manfaat"
    if "keluarga berencana" in t and "tujuan" in t:
        return "keluarga_berencana", "kb_tujuan"
    if "keluarga berencana" in t or "kb" in t:
        return "keluarga_berencana", "kb_pengertian"

    if "tumbuh kembang" in t and ("kognitif" in t or "bahasa" in t):
        return "tumbuh_kembang", "tumbuh_kembang_kognitif"
    if "tumbuh kembang" in t and "motorik" in t:
        return "tumbuh_kembang", "tumbuh_kembang_motorik"
    if "tumbuh kembang" in t or "berat badan" in t or "tinggi badan" in t:
        return "tumbuh_kembang", "tumbuh_kembang_fisik"

    if "balita" in t and "stimulasi" in t:
        return "balita", "balita_stimulasi"
    if "balita" in t and ("gizi" in t or "makan" in t):
        return "balita", "balita_nutrisi"
    if "balita" in t:
        return "balita", "balita_perkembangan"

    if "penyimpanan asi" in t or "asi perah" in t:
        return "menyusui", "menyusui_penyimpanan_asi"
    if "puting lecet" in t or "mastitis" in t or "masalah menyusui" in t:
        return "menyusui", "menyusui_masalah"
    if "teknik menyusui" in t or "perlekatan" in t or "posisi menyusui" in t:
        return "menyusui", "menyusui_teknik"
    if "manfaat asi" in t or "manfaat menyusui" in t:
        return "menyusui", "menyusui_manfaat"
    if "menyusui" in t or "air susu ibu" in t:
        return "menyusui", "menyusui_pengertian"

    return "kehamilan", "kehamilan_umum"


def classify_type_priority(title: str, content: str):
    t = f"{title} {content}".lower()
    if "tanda bahaya" in t:
        return "tanda_bahaya", "kritis"
    if "tidak boleh" in t or "hindari" in t or "larangan" in t:
        return "larangan", "penting"
    if "yang harus dilakukan" in t or "anjuran" in t or "disarankan" in t:
        return "anjuran", "penting"
    if "perkembangan" in t or "tumbuh kembang" in t:
        return "perkembangan", "umum"
    return "edukasi", "umum"


def extract_keywords(title: str, content: str, max_kw=8):
    text = re.sub(r"[^a-z0-9\s]", " ", (title + " " + content).lower())
    stop = {
        "yang", "dan", "atau", "dengan", "untuk", "pada", "dari", "ke", "di", "ini", "itu", "ibu",
        "anak", "bayi", "balita", "dalam", "adalah", "agar", "jika", "lebih", "setelah", "saat", "masa",
        "serta", "tidak", "boleh", "harus", "segera", "oleh", "sudah", "akan", "juga", "halaman",
    }
    freq = {}
    for w in text.split():
        if len(w) > 2 and w not in stop:
            freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:max_kw]]


def post_merge(records, merge_under=200, merge_max_total=500, split_min_words=200, split_max_words=500):
    out = []
    for rec in records:
        wc = len(rec["content"].split())
        if out and wc < merge_under and out[-1]["category"] == rec["category"]:
            prev_wc = len(out[-1]["content"].split())
            if prev_wc + wc <= merge_max_total:
                out[-1]["content"] += "\n\n" + rec["content"]
                out[-1]["keywords"] = list(dict.fromkeys(out[-1]["keywords"] + rec["keywords"]))[:10]
                if rec["priority"] == "kritis":
                    out[-1]["priority"] = "kritis"
                    out[-1]["type"] = "tanda_bahaya"
                continue
        out.append(rec)
    final = []
    for rec in out:
        wc = len(rec["content"].split())
        if wc <= split_max_words:
            final.append(rec)
            continue
        sents = re.split(r"(?<=[.!?])\s+", rec["content"])
        buf = []
        buf_w = 0
        part = 1
        for s in sents:
            sw = len(s.split())
            if buf_w + sw > split_max_words and buf_w >= split_min_words:
                c2 = dict(rec)
                c2["title"] = f"{rec['title']} - Bagian {part}"
                c2["content"] = " ".join(buf).strip()
                final.append(c2)
                part += 1
                buf = [s]
                buf_w = sw
            else:
                buf.append(s)
                buf_w += sw
        if buf:
            c2 = dict(rec)
            c2["title"] = f"{rec['title']} - Bagian {part}"
            c2["content"] = " ".join(buf).strip()
            final.append(c2)

    repaired = []
    for rec in final:
        wc = len(rec["content"].split())
        if repaired and wc < merge_under:
            prev_wc = len(repaired[-1]["content"].split())
            if repaired[-1]["category"] == rec["category"] and prev_wc + wc <= merge_max_total:
                repaired[-1]["content"] += "\n\n" + rec["content"]
                repaired[-1]["keywords"] = list(dict.fromkeys(repaired[-1]["keywords"] + rec["keywords"]))[:10]
                if rec["priority"] == "kritis":
                    repaired[-1]["priority"] = "kritis"
                    repaired[-1]["type"] = "tanda_bahaya"
                continue
        repaired.append(rec)
    return repaired


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


def main():
    parser = argparse.ArgumentParser(description="Build initial chunks from a maternal-child health PDF")
    parser.add_argument("--source", default=SOURCE_PDF, help="Source PDF file path")
    parser.add_argument("--out", default=OUT_FILE, help="Output JSON file path")
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=tuple(CHUNK_PROFILES.keys()),
        help="Chunking profile: strict|balanced|high_recall",
    )
    args = parser.parse_args()

    source_pdf = args.source
    out_file = args.out
    cfg = CHUNK_PROFILES[args.profile]

    pages_lines = extract_lines(source_pdf)
    sections = split_sections(pages_lines)

    records = []
    for sec_title, sec_content, sec_start_page, sec_content_type in sections:
        for part_idx, part in enumerate(split_special_blocks(sec_title, sec_content), start=1):
            chunks = chunk_text(
                part,
                min_words=cfg["min_words"],
                max_words=cfg["max_words"],
                min_output_words=cfg["min_output_words"],
            )
            for i, ch in enumerate(chunks, start=1):
                title = sec_title
                if len(chunks) > 1:
                    title = f"{sec_title} - Bagian {i}"
                if part_idx > 1:
                    title = f"{title} (Subtopik {part_idx})"

                cat, sub = classify_category_sub(title, ch)
                typ, pri = classify_type_priority(title, ch)
                rec = {
                    "title": title,
                    "content": ch,
                    "category": cat,
                    "sub_category": sub,
                    "type": typ,
                    "content_type": sec_content_type,
                    "start_page": sec_start_page,
                    "keywords": extract_keywords(title, ch),
                    "priority": pri,
                    "source": source_pdf,
                }
                rec["title"] = TITLE_MAP.get(sub, rec["title"])
                records.append(rec)

    final = post_merge(
        records,
        merge_under=cfg["merge_under"],
        merge_max_total=cfg["merge_max_total"],
        split_min_words=cfg["min_words"],
        split_max_words=cfg["max_words"],
    )
    Path(out_file).write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    lengths = [len(r["content"].split()) for r in final]
    print(f"Sections: {len(sections)}")
    print(f"Profile: {args.profile}")
    print(f"Chunks written: {len(final)}")
    if lengths:
        print(f"Word count min/max: {min(lengths)}/{max(lengths)}")


if __name__ == "__main__":
    main()
