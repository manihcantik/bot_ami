import json
import re
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_FILE = BASE_DIR / "hasil_chunking" / "kia_chunks.json"
OUTPUT_FILE = BASE_DIR / "hasil_chunking" / "kia_chunks_refined.json"
SOURCE = "Buku_KIA_2024 (1).pdf"

STOPWORDS = {
    "yang", "dan", "atau", "adalah", "untuk", "dengan", "pada", "dari", "ke", "di", "ini", "itu",
    "dalam", "agar", "jika", "serta", "oleh", "lebih", "setelah", "saat", "masa", "sudah", "akan", "juga",
    "seperti", "karena", "bisa", "dapat", "hal", "bagi", "kali", "hari", "tahun", "bulan", "ibu", "anak",
}

CATEGORY_SUBS = {
    "kehamilan": {
        "kehamilan_umum", "kehamilan_trimester_1", "kehamilan_trimester_2", "kehamilan_trimester_3",
        "kehamilan_nutrisi", "kehamilan_pemeriksaan", "kehamilan_aktivitas", "kehamilan_tanda_bahaya",
    },
    "persalinan": {
        "persalinan_persiapan", "persalinan_tanda_awal", "persalinan_proses", "persalinan_tanda_bahaya",
    },
    "nifas": {"nifas_perawatan", "nifas_pemulihan", "nifas_nutrisi", "nifas_tanda_bahaya"},
    "kesehatan_mental": {"mental_ibu_hamil", "mental_pasca_melahirkan", "mental_baby_blues", "mental_depresi"},
    "menyusui": {"menyusui_pengertian", "menyusui_manfaat", "menyusui_teknik", "menyusui_masalah", "menyusui_penyimpanan_asi"},
    "bayi_baru_lahir": {"bayi_baru_lahir_perawatan", "bayi_baru_lahir_tanda_bahaya", "bayi_baru_lahir_pemeriksaan"},
    "bayi_0_6_bulan": {"bayi_asi_eksklusif", "bayi_perkembangan", "bayi_perawatan", "bayi_tanda_bahaya"},
    "balita": {"balita_perkembangan", "balita_nutrisi", "balita_stimulasi"},
    "imunisasi": {"imunisasi_pengertian", "imunisasi_jadwal", "imunisasi_manfaat"},
    "nutrisi": {"nutrisi_ibu_hamil", "nutrisi_ibu_menyusui", "nutrisi_bayi", "nutrisi_balita"},
    "keluarga_berencana": {"kb_pengertian", "kb_tujuan", "kb_jenis", "kb_manfaat"},
    "tumbuh_kembang": {"tumbuh_kembang_fisik", "tumbuh_kembang_motorik", "tumbuh_kembang_kognitif"},
}


def normalize_spaces(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_page_and_clean(text: str) -> Tuple[str, int]:
    pages = [int(x) for x in re.findall(r"\[HALAMAN\s+(\d+)\]", text, flags=re.IGNORECASE)]
    page = pages[0] if pages else None
    text = re.sub(r"\[HALAMAN\s+\d+\]", "", text, flags=re.IGNORECASE)

    drop_patterns = [
        r"\bISBN\b", r"\bAPBN\b", r"\bCATATAN MEDIK\b", r"\bMOTHER\b", r"\bCHILD\b",
        r"\bKOHORT\b", r"\bNIK\b", r"\bNo\.\b", r"\bStamp\b", r"\bParaf\b",
    ]

    lines = [ln.strip() for ln in text.split("\n")]
    cleaned_lines = []
    for ln in lines:
        if not ln:
            continue
        up = ln.upper()
        if any(re.search(pat, up, flags=re.IGNORECASE) for pat in drop_patterns):
            continue
        if re.fullmatch(r"[A-Z0-9\-\s&/]{5,}", ln) and len(ln.split()) <= 8:
            continue
        if re.fullmatch(r"\d+(?:[.,]\d+)?", ln):
            continue
        if len(re.findall(r"\d", ln)) > 8 and len(ln.split()) < 10:
            continue
        cleaned_lines.append(ln)

    text = "\n".join(cleaned_lines)
    text = re.sub(r"\b(Tanggal|Paraf|Kader|Nakes|No\.|Cek|Lembar)\b(?:[^\n]{0,60})$", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"\b\d{1,3}\s+\d{1,3}\b", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    return normalize_spaces(text), page

def detect_type(segment: str) -> str:
    s = segment.lower()
    if "tanda bahaya" in s or any(k in s for k in ["perdarahan", "kejang", "sesak napas", "demam tinggi"]):
        return "tanda_bahaya"
    if any(k in s for k in ["tidak boleh", "hindari", "jangan", "dilarang"]):
        return "larangan"
    if any(k in s for k in ["yang harus dilakukan", "anjuran", "lakukan", "segera periksa", "pastikan", "minum"]):
        return "anjuran"
    if any(k in s for k in ["perkembangan", "tumbuh kembang", "yang akan dialami", "penanda perkembangan"]):
        return "perkembangan"
    return "edukasi"


def split_by_type(content: str) -> List[Tuple[str, str]]:
    # Split by explicit headings/phrases to isolate warning/recommendation/prohibition blocks.
    anchor_regex = r"(?i)(?=tanda bahaya|yang harus dilakukan|hal-hal yang tidak boleh|tidak boleh dilakukan|larangan|yang akan dialami)"
    parts = [p.strip() for p in re.split(anchor_regex, content) if p and p.strip()]

    if len(parts) <= 1:
        return [(detect_type(content), content)]

    segments = [(detect_type(p), p) for p in parts]

    merged: List[Tuple[str, str]] = []
    for t, seg in segments:
        if not merged:
            merged.append((t, seg))
            continue
        if len(seg.split()) < 80 and merged[-1][0] == t:
            merged[-1] = (t, merged[-1][1] + "\n\n" + seg)
        else:
            merged.append((t, seg))

    return merged

def classify_category_sub(content: str, old_cat: str, old_sub: str) -> Tuple[str, str]:
    s = content.lower()

    if "trimester 1" in s or "1-3 bulan" in s:
        return "kehamilan", "kehamilan_trimester_1"
    if "trimester 2" in s or "4-6 bulan" in s:
        return "kehamilan", "kehamilan_trimester_2"
    if "trimester 3" in s or "7-9 bulan" in s:
        return "kehamilan", "kehamilan_trimester_3"
    if "tanda bahaya" in s and any(k in s for k in ["hamil", "kehamilan", "trimester"]):
        return "kehamilan", "kehamilan_tanda_bahaya"

    if "persiapan melahirkan" in s:
        return "persalinan", "persalinan_persiapan"
    if "tanda awal" in s and "melahir" in s:
        return "persalinan", "persalinan_tanda_awal"
    if "proses melahirkan" in s or "persalinan" in s:
        return "persalinan", "persalinan_proses"
    if "tanda bahaya" in s and "melahir" in s:
        return "persalinan", "persalinan_tanda_bahaya"

    if "nifas" in s and "tanda bahaya" in s:
        return "nifas", "nifas_tanda_bahaya"
    if "nifas" in s and any(k in s for k in ["pemulihan", "perubahan tubuh"]):
        return "nifas", "nifas_pemulihan"
    if "nifas" in s and any(k in s for k in ["makan", "gizi", "nutrisi"]):
        return "nifas", "nifas_nutrisi"
    if "nifas" in s:
        return "nifas", "nifas_perawatan"

    if "baby blues" in s:
        return "kesehatan_mental", "mental_baby_blues"
    if "depresi" in s:
        return "kesehatan_mental", "mental_depresi"
    if "mental" in s or "stres" in s or "cemas" in s:
        return "kesehatan_mental", "mental_ibu_hamil"

    if "penyimpanan asi" in s or "asi perah" in s:
        return "menyusui", "menyusui_penyimpanan_asi"
    if "teknik menyusui" in s or "perlekatan" in s or ("posisi" in s and "menyusui" in s):
        return "menyusui", "menyusui_teknik"
    if "masalah menyusui" in s or "mastitis" in s or "puting lecet" in s:
        return "menyusui", "menyusui_masalah"
    if "manfaat asi" in s or "manfaat menyusui" in s:
        return "menyusui", "menyusui_manfaat"
    if "menyusui" in s or "air susu ibu" in s or "asi eksklusif" in s:
        return "menyusui", "menyusui_pengertian"
    if "bayi baru lahir" in s and "tanda bahaya" in s:
        return "bayi_baru_lahir", "bayi_baru_lahir_tanda_bahaya"
    if "bayi baru lahir" in s and any(k in s for k in ["pemeriksaan", "kunjungan"]):
        return "bayi_baru_lahir", "bayi_baru_lahir_pemeriksaan"
    if "bayi baru lahir" in s:
        return "bayi_baru_lahir", "bayi_baru_lahir_perawatan"

    if "0 - 6 bulan" in s and "tanda bahaya" in s:
        return "bayi_0_6_bulan", "bayi_tanda_bahaya"
    if "0 - 6 bulan" in s and "perkembangan" in s:
        return "bayi_0_6_bulan", "bayi_perkembangan"
    if "0 - 6 bulan" in s and "asi" in s:
        return "bayi_0_6_bulan", "bayi_asi_eksklusif"
    if "0 - 6 bulan" in s:
        return "bayi_0_6_bulan", "bayi_perawatan"

    if "balita" in s and "stimulasi" in s:
        return "balita", "balita_stimulasi"
    if "balita" in s and any(k in s for k in ["nutrisi", "gizi", "makan"]):
        return "balita", "balita_nutrisi"
    if "balita" in s:
        return "balita", "balita_perkembangan"

    if "imunisasi" in s and "jadwal" in s:
        return "imunisasi", "imunisasi_jadwal"
    if "imunisasi" in s and "manfaat" in s:
        return "imunisasi", "imunisasi_manfaat"
    if "imunisasi" in s:
        return "imunisasi", "imunisasi_pengertian"

    if "keluarga berencana" in s and "jenis" in s:
        return "keluarga_berencana", "kb_jenis"
    if "keluarga berencana" in s and "tujuan" in s:
        return "keluarga_berencana", "kb_tujuan"
    if "keluarga berencana" in s and "manfaat" in s:
        return "keluarga_berencana", "kb_manfaat"
    if "keluarga berencana" in s or "kontrasepsi" in s or re.search(r"\bkb\b", s):
        return "keluarga_berencana", "kb_pengertian"

    if "tumbuh kembang" in s and any(k in s for k in ["bahasa", "kognitif"]):
        return "tumbuh_kembang", "tumbuh_kembang_kognitif"
    if "tumbuh kembang" in s and "motorik" in s:
        return "tumbuh_kembang", "tumbuh_kembang_motorik"
    if "tumbuh kembang" in s or any(k in s for k in ["berat badan", "tinggi badan", "lingkar kepala"]):
        return "tumbuh_kembang", "tumbuh_kembang_fisik"

    # Keep old if it is valid.
    if old_cat in CATEGORY_SUBS and old_sub in CATEGORY_SUBS[old_cat]:
        return old_cat, old_sub

    return "kehamilan", "kehamilan_umum"


def priority_for_type(info_type: str) -> str:
    if info_type == "tanda_bahaya":
        return "kritis"
    if info_type in {"anjuran", "larangan"}:
        return "penting"
    return "umum"


def specific_title(cat: str, sub: str, info_type: str, content: str) -> str:
    base = {
        "kehamilan_trimester_1": "Kehamilan Trimester 1",
        "kehamilan_trimester_2": "Kehamilan Trimester 2",
        "kehamilan_trimester_3": "Kehamilan Trimester 3",
        "kehamilan_tanda_bahaya": "Kehamilan - Tanda Bahaya",
        "kehamilan_nutrisi": "Kehamilan - Nutrisi",
        "kehamilan_pemeriksaan": "Kehamilan - Pemeriksaan",
        "kehamilan_aktivitas": "Kehamilan - Aktivitas",
        "kehamilan_umum": "Kehamilan",
        "persalinan_persiapan": "Persalinan - Persiapan",
        "persalinan_tanda_awal": "Persalinan - Tanda Awal",
        "persalinan_proses": "Persalinan - Proses",
        "persalinan_tanda_bahaya": "Persalinan - Tanda Bahaya",
        "nifas_perawatan": "Masa Nifas - Perawatan",
        "nifas_pemulihan": "Masa Nifas - Pemulihan",
        "nifas_nutrisi": "Masa Nifas - Nutrisi",
        "nifas_tanda_bahaya": "Masa Nifas",
        "mental_ibu_hamil": "Kesehatan Mental Ibu Hamil",
        "mental_pasca_melahirkan": "Kesehatan Mental Pasca Melahirkan",
        "mental_baby_blues": "Baby Blues",
        "mental_depresi": "Depresi Pasca Melahirkan",
        "menyusui_pengertian": "Menyusui",
        "menyusui_manfaat": "Menyusui - Manfaat",
        "menyusui_teknik": "Menyusui - Teknik",
        "menyusui_masalah": "Menyusui - Masalah",
        "menyusui_penyimpanan_asi": "Menyusui - Penyimpanan ASI",
        "bayi_baru_lahir_perawatan": "Bayi Baru Lahir - Perawatan",
        "bayi_baru_lahir_tanda_bahaya": "Bayi Baru Lahir",
        "bayi_baru_lahir_pemeriksaan": "Bayi Baru Lahir - Pemeriksaan",
        "bayi_asi_eksklusif": "Bayi 0-6 Bulan - ASI Eksklusif",
        "bayi_perkembangan": "Bayi 0-6 Bulan - Perkembangan",
        "bayi_perawatan": "Bayi 0-6 Bulan - Perawatan",
        "bayi_tanda_bahaya": "Bayi 0-6 Bulan",
        "balita_perkembangan": "Balita - Perkembangan",
        "balita_nutrisi": "Balita - Nutrisi",
        "balita_stimulasi": "Balita - Stimulasi",
        "imunisasi_pengertian": "Imunisasi",
        "imunisasi_jadwal": "Imunisasi - Jadwal",
        "imunisasi_manfaat": "Imunisasi - Manfaat",
        "kb_pengertian": "Keluarga Berencana",
        "kb_tujuan": "Keluarga Berencana - Tujuan",
        "kb_jenis": "Keluarga Berencana - Jenis Metode",
        "kb_manfaat": "Keluarga Berencana - Manfaat",
        "tumbuh_kembang_fisik": "Tumbuh Kembang - Fisik",
        "tumbuh_kembang_motorik": "Tumbuh Kembang - Motorik",
        "tumbuh_kembang_kognitif": "Tumbuh Kembang - Kognitif",
    }.get(sub, "Edukasi Kesehatan Ibu dan Anak")

    c = content.lower()
    detail = None
    if info_type == "tanda_bahaya":
        if "perdarahan" in c:
            detail = "Tanda Bahaya Perdarahan"
        elif "demam" in c:
            detail = "Tanda Bahaya Demam"
        elif "sesak" in c:
            detail = "Tanda Bahaya Gangguan Napas"
        else:
            detail = "Tanda Bahaya"
    elif info_type == "anjuran":
        if "pemeriksaan" in c or "periksa" in c:
            detail = "Anjuran Pemeriksaan"
        elif "makan" in c or "gizi" in c:
            detail = "Anjuran Nutrisi"
        elif "asi" in c:
            detail = "Anjuran Pemberian ASI"
        else:
            detail = "Anjuran Perawatan"
    elif info_type == "larangan":
        if "merokok" in c:
            detail = "Larangan Merokok"
        elif "makanan" in c:
            detail = "Larangan Pola Makan"
        else:
            detail = "Larangan"
    elif info_type == "perkembangan":
        detail = "Tahap Perkembangan"

    return f"{base} - {detail}" if detail else base


def extract_keywords(content: str, max_kw: int = 10) -> List[str]:
    text = re.sub(r"[^a-z0-9\s-]", " ", content.lower())
    words = [w for w in text.split() if len(w) > 2 and w not in STOPWORDS and not w.isdigit()]
    freq: Dict[str, int] = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Boost medically relevant terms for RAG retrieval quality.
    boost_terms = [
        "demam", "perdarahan", "kejang", "sesak", "mual", "muntah", "diare", "nyeri", "asi", "imunisasi",
        "kehamilan", "trimester", "persalinan", "nifas", "bayi", "balita", "stunting", "gizi", "pemeriksaan",
        "kesehatan", "bahaya", "kontrasepsi", "kb", "perkembangan", "tumbuh", "menyusui",
    ]
    for t in boost_terms:
        if t in freq:
            freq[t] += 3

    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))
    return [k for k, _ in ranked[:max_kw]]


def clean_content_for_output(content: str) -> str:
    content = re.sub(r"\s+([,.;:!?])", r"\1", content)
    content = re.sub(r"([,.;:!?])(\w)", r"\1 \2", content)
    content = re.sub(r"\n{3,}", "\n\n", content)
    return content.strip()


def split_long_chunks(record: Dict, max_words: int = 500) -> List[Dict]:
    words = record["content"].split()
    if len(words) <= max_words:
        return [record]

    sentences = re.split(r"(?<=[.!?])\s+", record["content"])
    out: List[Dict] = []
    cur = []
    cur_w = 0
    idx = 1
    for s in sentences:
        sw = len(s.split())
        if cur_w + sw > max_words and cur_w >= 200:
            r = dict(record)
            r["title"] = f"{record['title']} (Bagian {idx})"
            r["content"] = " ".join(cur).strip()
            out.append(r)
            idx += 1
            cur = [s]
            cur_w = sw
        else:
            cur.append(s)
            cur_w += sw
    if cur:
        r = dict(record)
        r["title"] = f"{record['title']} (Bagian {idx})" if idx > 1 else record["title"]
        r["content"] = " ".join(cur).strip()
        out.append(r)
    return out


def merge_short_chunks(records: List[Dict], min_words: int = 120) -> List[Dict]:
    # Keep one topic/type by merging only same sub_category + type.
    out: List[Dict] = []
    for r in records:
        wc = len(r["content"].split())
        if out and wc < min_words:
            prev = out[-1]
            prev_wc = len(prev["content"].split())
            if prev["sub_category"] == r["sub_category"] and prev["type"] == r["type"] and prev_wc + wc <= 500:
                prev["content"] += "\n\n" + r["content"]
                prev["keywords"] = list(dict.fromkeys(prev["keywords"] + r["keywords"]))[:10]
                continue
        out.append(r)
    return out


def noise_score(text: str) -> float:
    words = text.split()
    if not words:
        return 1.0
    digit_tokens = sum(1 for w in words if re.search(r"\d", w))
    short_tokens = sum(1 for w in words if len(w) <= 2)
    return (digit_tokens + short_tokens) / len(words)


def refine() -> List[Dict]:
    raw = json.loads(INPUT_FILE.read_text(encoding="utf-8"))
    refined: List[Dict] = []

    for item in raw:
        content_raw = item.get("content", "")
        cleaned, page = extract_page_and_clean(content_raw)
        if len(cleaned.split()) < 30:
            continue

        typed_segments = split_by_type(cleaned)
        for seg_type, seg_content in typed_segments:
            seg_content = clean_content_for_output(seg_content)
            if len(seg_content.split()) < 20:
                continue
            if noise_score(seg_content) > 0.35:
                continue

            cat, sub = classify_category_sub(seg_content, item.get("category", ""), item.get("sub_category", ""))
            rec = {
                "chunk_id": "",
                "title": specific_title(cat, sub, seg_type, seg_content),
                "content": seg_content,
                "category": cat,
                "sub_category": sub,
                "type": seg_type,
                "keywords": extract_keywords(seg_content),
                "priority": priority_for_type(seg_type),
                "page": str(page) if page is not None else "",
                "source": SOURCE,
            }
            refined.extend(split_long_chunks(rec))

    refined = merge_short_chunks(refined, min_words=120)

    # Reassign chunk IDs.
    for i, r in enumerate(refined, start=1):
        r["chunk_id"] = f"KIA-{i:03d}"

    return refined

def main():
    global INPUT_FILE, OUTPUT_FILE, SOURCE

    parser = argparse.ArgumentParser(description="Refine chunk JSON into cleaner, typed chunks")
    parser.add_argument("--input", default=str(INPUT_FILE), help="Input JSON file path")
    parser.add_argument("--out", default=str(OUTPUT_FILE), help="Output JSON file path")
    parser.add_argument("--source", default=SOURCE, help="Source PDF file name/path metadata")
    args = parser.parse_args()

    INPUT_FILE = Path(args.input)
    OUTPUT_FILE = Path(args.out)
    SOURCE = args.source

    refined = refine()
    OUTPUT_FILE.write_text(json.dumps(refined, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Input chunks: {len(json.loads(INPUT_FILE.read_text(encoding='utf-8')))}")
    print(f"Refined chunks: {len(refined)}")


if __name__ == "__main__":
    main()
