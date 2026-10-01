import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "sumber_data"
CHUNK_DIR = PROJECT_ROOT / "data" / "chunks"
BUILD_SCRIPT = PROJECT_ROOT / "experiments" / "build_kia_chunks.py"


# ============================================================
# SKENARIO
# ============================================================

SCENARIOS = {
    "S1": {
        "chunk_size": 300,
        "overlap": 50,
        "top_k": 3,
        "desc": "Chunk kecil, k kecil",
    },
    "S2": {
        "chunk_size": 300,
        "overlap": 50,
        "top_k": 5,
        "desc": "Chunk kecil, k sedang",
    },
    "S3": {
        "chunk_size": 300,
        "overlap": 50,
        "top_k": 7,
        "desc": "Chunk kecil, k besar",
    },
    "S4": {
        "chunk_size": 500,
        "overlap": 50,
        "top_k": 3,
        "desc": "Chunk sedang, k kecil",
    },
    "S5": {
        "chunk_size": 500,
        "overlap": 50,
        "top_k": 5,
        "desc": "Chunk sedang, k sedang",
    },
    "S6": {
        "chunk_size": 500,
        "overlap": 50,
        "top_k": 7,
        "desc": "Chunk sedang, k besar",
    },
    "S7": {
        "chunk_size": 700,
        "overlap": 50,
        "top_k": 3,
        "desc": "Chunk besar, k kecil",
    },
    "S8": {
        "chunk_size": 700,
        "overlap": 50,
        "top_k": 5,
        "desc": "Chunk besar, k sedang",
    },
    "S9": {
        "chunk_size": 700,
        "overlap": 50,
        "top_k": 7,
        "desc": "Chunk besar, k besar",
    },
}


# ============================================================
# VALIDASI
# ============================================================

def validate_environment():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Folder sumber_data tidak ditemukan:\n{SOURCE_DIR}"
        )

    if not BUILD_SCRIPT.exists():
        raise FileNotFoundError(
            f"build_kia_chunks.py tidak ditemukan:\n{BUILD_SCRIPT}"
        )


# ============================================================
# AMBIL SEMUA PDF
# ============================================================

def get_source_files():
    """
    Mengambil semua file PDF yang berada langsung
    di dalam folder data/sumber_data.
    """

    pdf_files = sorted(
        [
            p
            for p in SOURCE_DIR.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".pdf"
        ],
        key=lambda p: p.name.lower(),
    )

    return pdf_files


# ============================================================
# NAMA FILE AMAN
# ============================================================

def safe_filename(name):
    """
    Membersihkan nama file agar aman digunakan
    sebagai nama file JSON.
    """

    name = re.sub(
        r'[<>:"/\\|?*]',
        "_",
        name,
    )

    name = re.sub(
        r"\s+",
        "_",
        name.strip(),
    )

    return name[:180].rstrip("._ ")


# ============================================================
# BUILD SATU PDF
# ============================================================

def build_one_pdf(
    scenario_id,
    source_pdf,
    output_json,
    chunk_size,
    overlap,
    profile,
):
    """
    Menjalankan build_kia_chunks.py untuk satu PDF.

    Satu PDF menghasilkan satu file JSON.
    """

    output_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    command = [
        sys.executable,
        str(BUILD_SCRIPT),

        "--source",
        str(source_pdf),

        "--profile",
        profile,

        "--chunk-size",
        str(chunk_size),

        "--overlap",
        str(overlap),

        "--out",
        str(output_json),
    ]

    print()
    print("-" * 70)
    print(f"SKENARIO : {scenario_id}")
    print(f"PDF      : {source_pdf.name}")
    print(f"Chunk    : {chunk_size}")
    print(f"Overlap  : {overlap}")
    print(f"Output   : {output_json.name}")
    print("-" * 70)

    result = subprocess.run(
        command,
        check=False,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    if stdout.strip():
        print(stdout.strip())

    # --------------------------------------------------------
    # DETEKSI ERROR PDF
    # --------------------------------------------------------

    errors = [
        line.strip()
        for line in stderr.splitlines()
        if (
            "MuPDF error" in line
            or "zlib error" in line
            or (
                "error" in line.lower()
                and (
                    "pdf" in line.lower()
                    or "mupdf" in line.lower()
                )
            )
        )
    ]

    if result.returncode != 0 or errors:

        if stderr.strip():
            print("PERINGATAN / ERROR:")
            print(stderr.strip())

        if output_json.exists():
            output_json.unlink()

        return {
            "success": False,
            "returncode": result.returncode,
            "errors": (
                errors
                or [
                    stderr.strip()
                    or "Build script gagal."
                ]
            ),
            "chunks": 0,
        }

    if stderr.strip():
        print(stderr.strip())

    # --------------------------------------------------------
    # CEK OUTPUT
    # --------------------------------------------------------

    if not output_json.exists():

        return {
            "success": False,
            "returncode": result.returncode,
            "errors": [
                "Output JSON tidak terbentuk."
            ],
            "chunks": 0,
        }

    # --------------------------------------------------------
    # VALIDASI JSON
    # --------------------------------------------------------

    try:

        data = json.loads(
            output_json.read_text(
                encoding="utf-8"
            )
        )

        if not isinstance(data, list):
            raise ValueError(
                "Format JSON harus berupa list."
            )

        chunk_count = len(data)

    except Exception as e:

        output_json.unlink(
            missing_ok=True
        )

        return {
            "success": False,
            "returncode": result.returncode,
            "errors": [
                f"JSON output tidak valid: {e}"
            ],
            "chunks": 0,
        }

    return {
        "success": True,
        "returncode": result.returncode,
        "errors": [],
        "chunks": chunk_count,
    }


# ============================================================
# BERSIHKAN OUTPUT LAMA
# ============================================================

def remove_old_scenario_outputs(scenario_dir):

    scenario_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for item in scenario_dir.iterdir():

        if (
            item.is_file()
            and item.suffix.lower() == ".json"
        ):
            item.unlink()

        elif (
            item.is_dir()
            and item.name == "_temp"
        ):
            shutil.rmtree(item)


# ============================================================
# SATU SKENARIO
# ============================================================

def run_single_scenario(
    scenario_id,
    profile="balanced",
):
    """
    Menjalankan satu skenario terhadap semua PDF.

    Setiap PDF menghasilkan SATU JSON.
    """

    scenario_id = scenario_id.upper().strip()

    if scenario_id not in SCENARIOS:
        raise ValueError(
            f"Skenario tidak ditemukan: {scenario_id}"
        )

    config = SCENARIOS[scenario_id]

    pdf_files = get_source_files()

    if not pdf_files:
        raise FileNotFoundError(
            f"Tidak ada file PDF di:\n{SOURCE_DIR}"
        )

    scenario_dir = (
        CHUNK_DIR
        / scenario_id
    )

    # Bersihkan output lama
    remove_old_scenario_outputs(
        scenario_dir
    )

    print()
    print("=" * 70)
    print(f"CHUNKING {scenario_id}")
    print("=" * 70)
    print(
        f"Deskripsi  : {config['desc']}"
    )
    print(
        f"Chunk Size : {config['chunk_size']}"
    )
    print(
        f"Overlap    : {config['overlap']}"
    )
    print(
        f"Top-K      : {config['top_k']}"
    )
    print(
        f"Jumlah PDF : {len(pdf_files)}"
    )
    print("=" * 70)

    generated_files = []
    failed_files = []
    total_chunks = 0

    # --------------------------------------------------------
    # PROSES SEMUA PDF
    # --------------------------------------------------------

    for index, pdf_file in enumerate(
        pdf_files,
        start=1,
    ):

        output_name = (
            f"{index:02d}_"
            f"{safe_filename(pdf_file.stem)}"
            f"_chunks.json"
        )

        output_json = (
            scenario_dir
            / output_name
        )

        print()
        print(
            f"[{index}/{len(pdf_files)}] "
            f"Memproses: {pdf_file.name}"
        )

        result = build_one_pdf(
            scenario_id=scenario_id,
            source_pdf=pdf_file,
            output_json=output_json,
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
            profile=profile,
        )

        if result["success"]:

            generated_files.append(
                output_json
            )

            total_chunks += result["chunks"]

            print(
                f"[OK] {pdf_file.name} "
                f"-> {result['chunks']} chunks"
            )

        else:

            failed_files.append(
                {
                    "file": pdf_file.name,
                    "errors": result["errors"],
                }
            )

            print(
                f"[GAGAL] {pdf_file.name} "
                f"-> dilewati."
            )

    # --------------------------------------------------------
    # HASIL SKENARIO
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"[OK] SKENARIO {scenario_id} SELESAI"
    )
    print("=" * 70)

    print(
        f"PDF ditemukan : {len(pdf_files)}"
    )

    print(
        f"PDF berhasil  : {len(generated_files)}"
    )

    print(
        f"PDF gagal     : {len(failed_files)}"
    )

    print(
        f"Output JSON   : {len(generated_files)}"
    )

    print(
        f"Total chunks  : {total_chunks}"
    )

    if failed_files:

        print(
            "STATUS        : PARTIAL"
        )

        print()
        print("PDF YANG GAGAL:")

        for item in failed_files:
            print(
                f"- {item['file']}"
            )

    else:

        print(
            "STATUS        : LENGKAP"
        )

    print("=" * 70)

    return {
        "scenario": scenario_id,
        "chunk_size": config["chunk_size"],
        "overlap": config["overlap"],
        "top_k": config["top_k"],
        "pdf_count": len(pdf_files),
        "pdf_success": len(generated_files),
        "pdf_failed": len(failed_files),
        "json_count": len(generated_files),
        "failed_files": failed_files,
        "chunk_count": total_chunks,
        "output_dir": str(
            scenario_dir
        ),
        "status": (
            "partial"
            if failed_files
            else "complete"
        ),
    }


# ============================================================
# SEMUA SKENARIO
# ============================================================

def run_all_scenarios(
    profile="balanced",
):
    """
    Menjalankan S1 sampai S9.

    Setiap skenario memproses semua PDF.
    """

    pdf_files = get_source_files()

    if not pdf_files:
        raise FileNotFoundError(
            f"Tidak ada file PDF di:\n{SOURCE_DIR}"
        )

    print()
    print("#" * 70)
    print(
        "# MENJALANKAN SEMUA SKENARIO S1-S9"
    )
    print("#" * 70)

    print(
        f"# Total PDF yang akan diproses: "
        f"{len(pdf_files)}"
    )

    print("#" * 70)

    print()
    print(
        "File PDF yang ditemukan:"
    )

    for index, pdf in enumerate(
        pdf_files,
        start=1,
    ):

        print(
            f"{index:02d}. {pdf.name}"
        )

    results = {}

    # --------------------------------------------------------
    # S1 - S9
    # --------------------------------------------------------

    for scenario_id in SCENARIOS:

        results[scenario_id] = (
            run_single_scenario(
                scenario_id=scenario_id,
                profile=profile,
            )
        )

    # --------------------------------------------------------
    # RINGKASAN
    # --------------------------------------------------------

    print()
    print("#" * 70)
    print(
        "# SEMUA SKENARIO SELESAI"
    )
    print("#" * 70)

    print()

    for (
        scenario_id,
        result,
    ) in results.items():

        print(
            f"{scenario_id} | "
            f"PDF={result['pdf_count']} | "
            f"JSON={result['json_count']} | "
            f"Chunks={result['chunk_count']} | "
            f"Chunk={result['chunk_size']} | "
            f"Overlap={result['overlap']} | "
            f"Top-K={result['top_k']} | "
            f"Status={result['status']}"
        )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Chunking seluruh PDF dalam "
            "data/sumber_data untuk "
            "eksperimen RAG S1-S9. "
            "Setiap PDF menghasilkan "
            "satu JSON per skenario."
        )
    )

    parser.add_argument(
        "--scenario",
        "-s",
        help=(
            "Jalankan satu skenario. "
            "Contoh: S5"
        ),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Jalankan seluruh skenario "
            "S1-S9 menggunakan semua PDF."
        ),
    )

    parser.add_argument(
        "--profile",
        default="balanced",
        choices=(
            "strict",
            "balanced",
            "high_recall",
        ),
        help="Profile chunking.",
    )

    args = parser.parse_args()

    if args.all and args.scenario:

        parser.error(
            "Gunakan --all ATAU --scenario, "
            "jangan keduanya."
        )

    # --------------------------------------------------------
    # VALIDASI
    # --------------------------------------------------------

    validate_environment()

    # --------------------------------------------------------
    # SATU SKENARIO
    # --------------------------------------------------------

    if args.scenario:

        result = run_single_scenario(
            scenario_id=args.scenario,
            profile=args.profile,
        )

        print()
        print("Hasil:")

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    # --------------------------------------------------------
    # SEMUA SKENARIO
    # --------------------------------------------------------

    if args.all:

        results = run_all_scenarios(
            profile=args.profile
        )

        summary_path = (
            PROJECT_ROOT
            / "results"
            / "chunking_summary.json"
        )

        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        summary_path.write_text(
            json.dumps(
                results,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print()
        print(
            "[OK] Summary tersimpan di:"
        )

        print(summary_path)

        return

    # --------------------------------------------------------
    # TIDAK ADA ARGUMEN
    # --------------------------------------------------------

    parser.print_help()

    print()
    print("Contoh:")

    print(
        "python .\\experiments\\chunking.py "
        "--scenario S5"
    )

    print(
        "python .\\experiments\\chunking.py "
        "--all"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()