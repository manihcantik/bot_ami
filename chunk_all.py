from pathlib import Path
import subprocess
import re
import sys

SOURCE_DIR = Path("sumber_data")
OUTPUT_DIR = Path("hasil_chunking")
OUTPUT_DIR.mkdir(exist_ok=True)

print(f"Menggunakan Python: {sys.executable}\n")

for pdf in sorted(SOURCE_DIR.glob("*.pdf")):
    nama = pdf.stem.lower()
    nama = re.sub(r"[^a-z0-9]+", "_", nama).strip("_")

    output = OUTPUT_DIR / f"{nama}_chunks_balanced.json"

    print("=" * 70)
    print(f"Memproses : {pdf.name}")

    result = subprocess.run([
        sys.executable,                     
        "kode_chunking/build_kia_chunks.py",
        "--source", str(pdf),
        "--profile", "balanced",
        "--out", str(output)
    ])

    if result.returncode != 0:
        print(f"Gagal memproses {pdf.name}")

print("\n Semua PDF selesai diproses.")