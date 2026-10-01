"""

Centralized, deterministic guardrails for the KIA education chatbot.

Alur scope:

    Jelas KIA

        -> ALLOW -> RAG

    Jelas bukan KIA

        -> REJECT -> Guardrail

    Tidak jelas / ambigu

        -> CLARIFY -> meminta pengguna memperjelas

Guardrail bekerja sebelum proses retrieval.

"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from pathlib import Path

# ============================================================

# 1. KONFIGURASI

# ============================================================

MAX_INPUT_CHARS = 2000

MIN_SIMILARITY = 0.25

MAX_CONTEXT_CHARS = 2000

# ============================================================

# 2. PESAN SISTEM

# ============================================================

DISCLAIMER = (

    "Disclaimer: Informasi ini bersifat edukatif dan bukan "

    "pengganti konsultasi dengan tenaga kesehatan."

)

EMERGENCY_DISCLAIMER = (

    "Segera hubungi layanan darurat 119 atau fasilitas "

    "kesehatan terdekat. Jangan menunda pertolongan."

)

NO_CONTEXT_MESSAGE = (

    "Maaf, informasi yang cukup relevan belum ditemukan "

    "dalam basis pengetahuan untuk menjawab pertanyaan ini. "

    "Silakan konsultasikan dengan tenaga kesehatan."

)

OUT_OF_SCOPE_MESSAGE = (

    "Maaf, saya hanya dapat membantu edukasi Kesehatan "

    "Ibu dan Anak (KIA). Silakan ajukan pertanyaan seputar "

    "kehamilan, persalinan, nifas, bayi, anak, imunisasi, "

    "atau gizi ibu dan anak."

)

CLARIFY_MESSAGE = (

    "Mohon jelaskan pertanyaan yang berkaitan dengan "

    "kesehatan ibu dan anak agar saya dapat membantu "

    "berdasarkan sumber yang tersedia."

)

# ============================================================

# 3. TOPIK KIA

# ============================================================

"""

Kumpulan indikator topik KIA.

Keyword dibagi berdasarkan konteks:

- KIA umum

- Kehamilan

- Persalinan

- Nifas

- Bayi

- Anak dan tumbuh kembang

- Imunisasi

- Gizi

- Keluhan kesehatan bayi/anak

- Pemeriksaan dan pelayanan KIA

- Frasa KIA yang kemungkinan digunakan

  secara tidak langsung oleh pengguna

"""

KIA_KEYWORDS = (

    # --------------------------------------------------------

    # KIA UMUM

    # --------------------------------------------------------

    "kia",

    "kesehatan ibu dan anak",

    "ibu dan anak",

    "kesehatan ibu",

    "kesehatan anak",

    # --------------------------------------------------------

    # KEHAMILAN

    # --------------------------------------------------------

    "hamil",

    "kehamilan",

    "ibu hamil",

    "janin",

    "kandungan",

    "trimester",

    "trimester pertama",

    "trimester kedua",

    "trimester ketiga",

    "usia kehamilan",

    "kehamilan muda",

    "kehamilan tua",

    "antenatal",

    "antenatal care",

    "anc",

    "pemeriksaan kehamilan",

    "periksa kehamilan",

    "kontrol kehamilan",

    "kontrol kandungan",

    "pemeriksaan kandungan",

    "periksa kandungan",

    "dokter kandungan",

    "bidan kehamilan",

    # --------------------------------------------------------

    # PERSALINAN

    # --------------------------------------------------------

    "persalinan",

    "melahirkan",

    "mau melahirkan",

    "akan melahirkan",

    "proses melahirkan",

    "proses persalinan",

    "melahirkan normal",

    "persalinan normal",

    "persalinan caesar",

    "operasi caesar",

    "operasi sesar",

    "sesar",

    "kontraksi",

    "kontraksi persalinan",

    "pembukaan persalinan",

    "tanda persalinan",

    "tanda melahirkan",

    "menjelang persalinan",

    "sebelum melahirkan",

    # --------------------------------------------------------

    # NIFAS

    # --------------------------------------------------------

    "nifas",

    "ibu nifas",

    "masa nifas",

    "pasca melahirkan",

    "pasca persalinan",

    "setelah melahirkan",

    "setelah persalinan",

    "habis melahirkan",

    "baru melahirkan",

    "pemulihan setelah melahirkan",

    "pemulihan pasca melahirkan",

    # --------------------------------------------------------

    # BAYI

    # --------------------------------------------------------

    "bayi",

    "bayi baru lahir",

    "bayi baru lahir",

    "bayi lahir",

    "neonatus",

    "newborn",

    "si kecil",

    "bayiku",

    "bayi saya",

    "bayi saya",

    "bayinya",

    # ASI dan menyusui

    "asi",

    "asi eksklusif",

    "air susu ibu",

    "menyusui",

    "menyusui bayi",

    "ibu menyusui",

    "kolostrum",

    "produksi asi",

    "asi ibu",

    # MPASI

    "mpasi",

    "mp asi",

    "makanan pendamping asi",

    "makanan pendamping",

    "pemberian mpasi",

    "pemberian makanan bayi",

    # --------------------------------------------------------

    # ANAK DAN TUMBUH KEMBANG

    # --------------------------------------------------------

    "balita",

    "bayi dan anak",

    "tumbuh kembang",

    "tumbuh kembang anak",

    "tumbuh kembang bayi",

    "pertumbuhan anak",

    "pertumbuhan bayi",

    "perkembangan anak",

    "perkembangan bayi",

    "perkembangan balita",

    "berat badan anak",

    "tinggi badan anak",

    "berat badan bayi",

    "tinggi badan bayi",

    "berat badan balita",

    "tinggi badan balita",

    "lingkar kepala",

    "lingkar lengan",

    # Indikator perkembangan yang sering ditanyakan

    "belum bisa duduk",

    "belum bisa berjalan",

    "belum bisa merangkak",

    "belum bisa tengkurap",

    "belum bisa berdiri",

    "belum bisa bicara",

    "belum tumbuh gigi",

    "perkembangan motorik",

    "motorik anak",

    "motorik bayi",

    # --------------------------------------------------------

    # IMUNISASI

    # --------------------------------------------------------

    "imunisasi",

    "vaksin",

    "vaksinasi",

    "imunisasi bayi",

    "imunisasi anak",

    "imunisasi balita",

    "jadwal imunisasi",

    "jadwal vaksin",

    "jenis imunisasi",

    "jenis vaksin",

    "vaksin bayi",

    "vaksin anak",

    "vaksin balita",

    # --------------------------------------------------------

    # GIZI

    # --------------------------------------------------------

    "gizi",

    "gizi ibu",

    "gizi anak",

    "gizi bayi",

    "gizi balita",

    "gizi ibu hamil",

    "gizi ibu menyusui",

    "nutrisi",

    "nutrisi ibu",

    "nutrisi anak",

    "nutrisi bayi",

    "nutrisi balita",

    "makanan ibu hamil",

    "makanan ibu menyusui",

    "makanan bayi",

    "makanan anak",

    "makanan balita",

    "stunting",

    "gizi buruk",

    "gizi kurang",

    "kurang gizi",

    "status gizi",

    # --------------------------------------------------------

    # KESEHATAN BAYI / ANAK

    # --------------------------------------------------------

    "demam pada bayi",

    "demam pada anak",

    "demam bayi",

    "demam anak",

    "batuk pada bayi",

    "batuk pada anak",

    "batuk bayi",

    "batuk anak",

    "diare pada bayi",

    "diare pada anak",

    "diare bayi",

    "diare anak",

    "sembelit pada anak",

    "sembelit pada bayi",

    "sakit perut pada anak",

    "sakit perut pada bayi",

    "anak sulit makan",

    "bayi sulit makan",

    "anak susah makan",

    "bayi susah makan",

    "anak tidak mau makan",

    "bayi tidak mau menyusu",

    # --------------------------------------------------------

    # PEMERIKSAAN / PELAYANAN KIA

    # --------------------------------------------------------

    "puskesmas",

    "posyandu",

    "posyandu balita",

    "posyandu ibu hamil",

    "buku kia",

    "buku kesehatan ibu dan anak",

    "pemeriksaan bayi",

    "pemeriksaan anak",

    "pemeriksaan ibu",

    "pelayanan kia",

    "pelayanan kesehatan ibu",

    "pelayanan kesehatan anak",
)

# Pola konteks KIA yang tidak selalu menyebut kata "kesehatan".
KIA_CONTEXT_PATTERNS = (
    r"\banak\s+usia\s+\d+\s*(?:[-–]\s*\d+)?\s*(?:bulan|tahun)\b",
    r"\bbayi\s+usia\s+\d+\s*(?:[-–]\s*\d+)?\s*(?:bulan|tahun)\b",
    r"\busia\s+\d+\s*(?:[-–]\s*\d+)?\s*(?:bulan|tahun)\b",
    r"\b(?:vitamin\s*a)\b",
    r"\b(?:tanda\s+bahaya|bahaya)\b.*\b(?:anak|bayi|balita|ibu|hamil|nifas|persalinan)\b",
    r"\b(?:anak|bayi|balita)\b.*\b(?:tanda\s+bahaya|bahaya)\b",
    r"\b(?:pelayanan|layanan|pemeriksaan)\b.*\b(?:anak|bayi|balita|ibu|hamil)\b",
    r"\b(?:anak|bayi|balita|ibu|hamil)\b.*\b(?:pelayanan|layanan|pemeriksaan)\b",
    r"\bkelas\s+ibu\s+balita\b",
    r"\bposyandu\b.*\b(?:anak|bayi|balita)\b",
    r"\b(?:stimulasi|perkembangan|pertumbuhan)\b.*\b(?:anak|bayi|balita)\b",
    r"\b(?:anak|bayi|balita)\b.*\b(?:stimulasi|perkembangan|pertumbuhan)\b",
    r"\b(?:mpasi|asi|imunisasi|vaksin)\b",
)

# ============================================================

# 4. EMERGENCY

# ============================================================

EMERGENCY_KEYWORDS = (

    "kejang",

    "pendarahan",

    "perdarahan",

    "tidak sadar",

    "sesak napas",

    "sesak nafas",

    "biru pada bayi",

    "bayi biru",

    "sulit bernapas",

    "sulit bernafas",

    "pingsan",

    "nyeri dada",

    "air ketuban pecah",

)

# ============================================================

# 5. DIAGNOSIS

# ============================================================

DIAGNOSIS_PATTERNS = (

    r"\b(?:saya|aku|ibu|bayi|anak)\s+"

    r"(?:kena|terkena|menderita)\b",

    r"\b(?:ini|gejala ini|keluhan ini)\s+"

    r"(?:penyakit|diagnosis|diagnosa)\s+apa\b",

    r"\b(?:diagnosis|diagnosa)\s*"

    r"(?:saya|aku|ini|nya)?\b",

    r"\bpenyakit apa\b",

    r"\bini penyakit apa\b",

    r"\bini termasuk penyakit apa\b",

)

# ============================================================

# 6. OBAT / DOSIS

# ============================================================

MEDICATION_PATTERNS = (

    r"\b(?:obat|merek obat|dosis|berapa mg|"

    r"berapa tablet|minum apa|pil)\b",

    r"\bobat apa\b",

    r"\bobat untuk\b",

    r"\bberapa dosis\b",

    r"\bberapa kali minum\b",

)

# ============================================================

# 7. TINDAKAN MEDIS

# ============================================================

MEDICAL_ACTION_PATTERNS = (

    r"\b(?:harus|perlu)\s+"

    r"(?:operasi|tindakan medis|rawat inap)\b",

    r"\btindakan medis\s+apa\b",

    r"\bperlu dirawat\b",

    r"\bharus dirawat\b",

)

# ============================================================

# 8. PROMPT INJECTION / ROLE OVERRIDE

# ============================================================

ROLE_OVERRIDE_PATTERNS = (

    r"abaikan\s+(?:semua\s+)?instruksi",

    r"anggap\s+kamu\s+(?:dokter|bidan)",

    r"jadilah\s+(?:dokter|bidan)",

    r"tampilkan\s+(?:system prompt|system\s+prompt|"

    r"instruksi sistem)",

    r"bocorkan\s+(?:system prompt|instruksi sistem)",

    r"abaikan\s+aturan",

    r"abaikan\s+guardrail",

    r"jangan ikuti instruksi",

)

UNSAFE_SCOPE_PATTERNS = (
    r"\b(?:mengobati|obati|merawat)\s+(?:anak|bayi|sendiri)\b",
    r"\btidak\s+perlu\s+ke\s+(?:puskesmas|dokter|rumah sakit)\b",
    r"\b(?:mengarang|membuat[- ]buat)\b.*\b(?:jawaban|informasi)\b",
)

# ============================================================

# 9. POLA NON-KIA YANG JELAS

# ============================================================

"""

Kategori ini digunakan untuk pertanyaan yang secara jelas

berada di luar ruang lingkup chatbot KIA.

Penting:

kata yang terlalu umum tidak dimasukkan.

"""

OUT_OF_SCOPE_PATTERNS = (

    # Politik

    r"\b(?:politik|pemilu|presiden|wakil presiden|"

    r"partai politik|kampanye|calon presiden|caleg)\b",

    # Pemrograman / teknologi

    r"\b(?:program komputer|kode program|coding|python|"

    r"javascript|java|html|css|php|flutter|dart|"

    r"laravel|database|sql|pemrograman)\b",

    # Pendidikan umum

    r"\b(?:matematika|fisika|kimia|sejarah|geografi|"

    r"ekonomi|akuntansi|pelajaran sekolah)\b",

    # Hiburan / olahraga

    r"\b(?:sepak bola|basket|voli|olahraga|film|musik|"

    r"game|permainan|artis|lagu|drama)\b",

    # Akademik

    r"\b(?:makalah|skripsi|proposal penelitian|"

    r"tugas kuliah|tugas sekolah)\b",

    # Informasi umum

    r"\b(?:cuaca|berita|nilai tukar|saham|investasi)\b",

    # Lokasi / wisata

    r"\b(?:wisata|hotel|restoran|tempat liburan)\b",

    # Kuliner umum di luar edukasi gizi KIA

    r"\b(?:resep|memasak|masakan|nasi goreng|kue)\b",

)

# ============================================================

# 10. AMBIGUOUS

# ============================================================

"""

Pertanyaan yang terlalu pendek atau tidak memberikan

konteks yang cukup.

"""

AMBIGUOUS_PATTERNS = (

    r"^(?:bagaimana)\\??$",

    r"^(?:tolong bantu)\\??$",

    r"^(?:ini kenapa)\\??$",

    r"^(?:apa itu)\\??$",

    r"^(?:kenapa)\\??$",

    r"^(?:mengapa)\\??$",

    r"^(?:normal nggak)\\??$",

    r"^(?:normal tidak)\\??$",

    r"^(?:boleh nggak)\\??$",

    r"^(?:boleh tidak)\\??$",

    r"^(?:aman nggak)\\??$",

    r"^(?:aman tidak)\\??$",

    r"^(?:yang mana)\\??$",

    r"^(?:bagaimana caranya)\\??$",

)

# ============================================================

# 11. INCOMPLETE

# ============================================================

INCOMPLETE_PATTERNS = (

    r"^(?:berapa)\\??$",

    r"^(?:kapan)\\??$",

    r"^(?:di mana|dimana)\\??$",

    r"^(?:boleh)\\??$",

    r"^(?:aman)\\??$",

    r"^(?:kenapa ini)\\??$",

)

# ============================================================

# 12. STRUKTUR KEPUTUSAN

# ============================================================

@dataclass

class GuardrailDecision:

    action: str = "allow"

    reason: str = ""

    response: str = ""

    flags: list[str] = field(

        default_factory=list

    )

# ============================================================

# 13. HELPER

# ============================================================

def _matches_any(

    text: str,

    patterns: tuple[str, ...]

) -> bool:

    return any(

        re.search(

            pattern,

            text,

            flags=re.IGNORECASE

        )

        for pattern in patterns

    )

def _contains_kia_topic(

    text: str

) -> bool:

    """

    Memeriksa apakah pertanyaan memiliki

    indikasi topik KIA.

    Pemeriksaan menggunakan keyword dan

    frasa yang sudah ditentukan.

    """

    normalized = " ".join(

        text.lower().split()

    )

    return (
        any(
            re.search(
                rf"(?<!\w){re.escape(keyword)}(?!\w)",
                normalized
            )
            for keyword in KIA_KEYWORDS
        )
        or _matches_any(
            normalized,
            KIA_CONTEXT_PATTERNS
        )
    )

def _is_clear_non_kia(

    text: str

) -> bool:

    """

    Memeriksa apakah pertanyaan secara jelas

    merupakan pertanyaan di luar KIA.

    """

    return _matches_any(

        text,

        OUT_OF_SCOPE_PATTERNS

    )

def _is_ambiguous(

    text: str

) -> bool:

    """

    Memeriksa pertanyaan yang terlalu pendek

    atau tidak mempunyai konteks yang cukup.

    """

    return _matches_any(

        text,

        AMBIGUOUS_PATTERNS

    )

def _is_incomplete(

    text: str

) -> bool:

    """

    Memeriksa pertanyaan yang belum lengkap.

    """

    return _matches_any(

        text,

        INCOMPLETE_PATTERNS

    )

# ============================================================

# 14. INSPECT QUERY

# ============================================================

def inspect_query(

    query: str

) -> GuardrailDecision:

    """

    Pemeriksaan guardrail sebelum retrieval.

    Alur utama:

    1. Input kosong

    2. Input terlalu panjang

    3. Emergency

    4. Role/prompt override

    5. Diagnosis

    6. Obat/dosis

    7. Tindakan medis

    8. Jelas bukan KIA

    9. Ambiguous / incomplete

    10. Jelas KIA

    11. Tidak jelas -> clarify

    Hasil akhir:

        Jelas KIA

            -> allow

        Jelas bukan KIA

            -> reject

        Tidak jelas / ambigu

            -> clarify

    """

    normalized = " ".join(

        query.lower().split()

    )

    # ========================================================

    # 1. INPUT KOSONG

    # ========================================================

    if not normalized:

        return GuardrailDecision(

            action="clarify",

            reason="empty_input",

            response=(

                "Silakan masukkan pertanyaan "

                "terlebih dahulu."

            ),

        )

    # ========================================================

    # 2. INPUT TERLALU PANJANG

    # ========================================================

    if len(query) > MAX_INPUT_CHARS:

        return GuardrailDecision(

            action="reject",

            reason="input_too_long",

            response=(

                "Pertanyaan terlalu panjang. "

                "Silakan ringkas pertanyaan Anda."

            ),

        )

    # ========================================================

    # 3. EMERGENCY

    # ========================================================

    if any(

        keyword in normalized

        for keyword in EMERGENCY_KEYWORDS

    ):

        return GuardrailDecision(

            action="emergency",

            reason="emergency_keyword",

            response=EMERGENCY_DISCLAIMER,

        )

    # ========================================================

    # 4. ROLE OVERRIDE / PROMPT INJECTION

    # ========================================================

    if _matches_any(

        normalized,

        ROLE_OVERRIDE_PATTERNS

    ):

        return GuardrailDecision(

            action="reject",

            reason="role_override_or_prompt_leak",

            response=(

                "Maaf, saya tidak dapat mengikuti "

                "permintaan tersebut. Saya tetap "

                "berperan sebagai chatbot edukasi KIA."

            ),

        )

    if _matches_any(
        normalized,
        UNSAFE_SCOPE_PATTERNS
    ):
        return GuardrailDecision(
            action="referral",
            reason="unsafe_or_unsupported_request",
            response=(
                "Saya tidak boleh mengarang informasi atau "
                "menyarankan pengobatan sendiri. Gunakan "
                "informasi dari Buku KIA dan hubungi tenaga "
                "kesehatan bila anak atau bayi mengalami keluhan."
            ),
        )

    # ========================================================

    # 5. DIAGNOSIS

    # ========================================================

    if _matches_any(

        normalized,

        DIAGNOSIS_PATTERNS

    ):

        return GuardrailDecision(

            action="referral",

            reason="diagnosis_request",

            response=(

                "Saya tidak dapat menentukan diagnosis "

                "dari percakapan. Untuk penilaian kondisi "

                "Anda, silakan hubungi tenaga kesehatan."

            ),

        )

    # ========================================================

    # 6. OBAT / DOSIS

    # ========================================================

    if _matches_any(

        normalized,

        MEDICATION_PATTERNS

    ):

        return GuardrailDecision(

            action="referral",

            reason="medication_or_dosage_request",

            response=(

                "Saya tidak dapat menyebutkan obat atau "

                "dosis tertentu. Silakan konsultasikan "

                "pilihan yang sesuai kepada tenaga kesehatan."

            ),

        )

    # ========================================================

    # 7. TINDAKAN MEDIS

    # ========================================================

    if _matches_any(

        normalized,

        MEDICAL_ACTION_PATTERNS

    ):

        return GuardrailDecision(

            action="referral",

            reason="unsupported_medical_action",

            response=(

                "Saya tidak dapat menentukan tindakan medis "

                "untuk kondisi tertentu. Silakan berkonsultasi "

                "langsung dengan tenaga kesehatan."

            ),

        )

    # ========================================================

    # 8. JELAS BUKAN KIA

    # ========================================================

    """

    Contoh:

        "Bagaimana membuat program Python?"

        "Siapa presiden Indonesia?"

        "Berapa hasil matematika ini?"

        "Film apa yang bagus?"

    Pertanyaan seperti ini jelas berada di luar

    ruang lingkup chatbot KIA.

    """

    if _is_clear_non_kia(

        normalized

    ):

        return GuardrailDecision(

            action="reject",

            reason="out_of_scope",

            response=OUT_OF_SCOPE_MESSAGE,

        )

    # ========================================================

    # 9. AMBIGUOUS

    # ========================================================

    """

    Pertanyaan yang terlalu pendek ditangani

    sebagai CLARIFY.

    Contoh:

        "Bagaimana?"

        "Ini kenapa?"

        "Normal nggak?"

        "Yang mana?"

    """

    if _is_ambiguous(

        normalized

    ):

        return GuardrailDecision(

            action="clarify",

            reason="ambiguous_question",

            response=CLARIFY_MESSAGE,

        )

    # ========================================================

    # 10. INCOMPLETE

    # ========================================================

    if _is_incomplete(

        normalized

    ):

        return GuardrailDecision(

            action="clarify",

            reason="incomplete_question",

            response=CLARIFY_MESSAGE,

        )

    # ========================================================

    # 11. JELAS KIA

    # ========================================================

    """

    Kalau terdapat indikator KIA yang jelas,

    pertanyaan diperbolehkan masuk ke RAG.

    Contoh:

        "Apa jadwal imunisasi bayi?"

        "Apa tanda persalinan?"

        "Apa itu ASI eksklusif?"

        "Berapa kali pemeriksaan kehamilan?"

        "Apa itu masa nifas?"

    """

    if _contains_kia_topic(

        normalized

    ):

        return GuardrailDecision(

            action="allow",

            reason="kia_scope",

            response="",

            flags=["kia_scope"],

        )

    # ========================================================

    # 12. TIDAK JELAS / TIDAK TERKLASIFIKASI

    # ========================================================

    """

    Ini bagian penting.

    Kalau pertanyaan:

    - bukan jelas KIA,

    - bukan jelas non-KIA,

    - bukan emergency,

    - bukan diagnosis,

    - bukan obat,

    - bukan tindakan medis,

    - dan bukan pertanyaan yang sudah jelas ambigu,

    maka JANGAN langsung reject.

    Pertanyaan diarahkan ke CLARIFY agar pengguna

    memberikan konteks tambahan.

    """

    return GuardrailDecision(

        action="clarify",

        reason="uncertain_scope",

        response=CLARIFY_MESSAGE,

        flags=["uncertain_scope"],

    )

# ============================================================

# 15. DISCLAIMER

# ============================================================

def append_disclaimer(

    answer: str,

    sensitive: bool = False

) -> str:

    """

    Memastikan setiap jawaban yang ditampilkan

    memiliki disclaimer.

    """

    answer = answer.strip()

    if DISCLAIMER not in answer:

        answer = (

            f"{answer}\n\n"

            f"{DISCLAIMER}"

        )

    if (

        sensitive

        and EMERGENCY_DISCLAIMER not in answer

    ):

        answer = (

            f"{answer}\n"

            f"{EMERGENCY_DISCLAIMER}"

        )

    return answer

# ============================================================

# 16. FORMAT SOURCES

# ============================================================

def format_sources(

    relevant_docs: list[tuple]

) -> str:

    """

    Format metadata sumber dokumen

    untuk ditampilkan sebagai sumber jawaban.

    """

    citations = []

    seen = set()

    for _, metadata, _, _ in relevant_docs:

        metadata = metadata or {}

        source = (

            metadata.get("source")

            or metadata.get("title")

            or "Dokumen KIA"

        )

        page = metadata.get(
            "page"
        ) or metadata.get(
            "start_page"
        )

        source_text = str(source)
        if source_text not in {"Dokumen KIA", ""}:
            source_text = Path(source_text).name

        citation = (
            f"{source_text}, halaman {page}"
            if page
            else source_text
        )

        if citation not in seen:

            citations.append(

                citation

            )

            seen.add(

                citation

            )

    return "; ".join(

        citations

    )