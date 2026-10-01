import argparse
import json
import chromadb
import hashlib
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.config import EMBEDDING_MODEL

# ============================================================
# SKENARIO
# ============================================================

SCENARIOS = {
    "S1": {"chunk_size": 300, "overlap": 50, "top_k": 3},
    "S2": {"chunk_size": 300, "overlap": 50, "top_k": 5},
    "S3": {"chunk_size": 300, "overlap": 50, "top_k": 7},
    "S4": {"chunk_size": 500, "overlap": 50, "top_k": 3},
    "S5": {"chunk_size": 500, "overlap": 50, "top_k": 5},
    "S6": {"chunk_size": 500, "overlap": 50, "top_k": 7},
    "S7": {"chunk_size": 700, "overlap": 50, "top_k": 3},
    "S8": {"chunk_size": 700, "overlap": 50, "top_k": 5},
    "S9": {"chunk_size": 700, "overlap": 50, "top_k": 7},
}

# ChromaDB memiliki batas batch. Gunakan batch kecil agar aman.
CHROMA_BATCH_SIZE = 1000
EMBED_BATCH_SIZE = 32

# ============================================================
# ARGUMENT
# ============================================================

parser = argparse.ArgumentParser(
    description="Embedding chunk JSON ke ChromaDB untuk eksperimen RAG S1-S9"
)

parser.add_argument(
    "--scenario",
    "-s",
    choices=list(SCENARIOS.keys()),
    help="Embed satu skenario, contoh: S5",
)

parser.add_argument(
    "--all",
    action="store_true",
    help="Embed seluruh skenario S1-S9",
)

parser.add_argument(
    "--input",
    help="Satu file JSON hasil chunking",
)

parser.add_argument(
    "--files",
    nargs="*",
    help="Beberapa file JSON yang akan di-embed",
)

parser.add_argument(
    "--dir",
    help="Directory berisi file JSON chunk",
)

parser.add_argument(
    "--chroma_path",
    "--chroma-path",
    help="Lokasi ChromaDB",
)

parser.add_argument(
    "--top_k",
    "--top-k",
    type=int,
    help="Nilai Top-K skenario",
)

parser.add_argument(
    "--chunk_size",
    "--chunk-size",
    type=int,
    help="Ukuran chunk",
)

parser.add_argument(
    "--overlap",
    type=int,
    help="Overlap chunk",
)

parser.add_argument(
    "--reset",
    action="store_true",
    help="Hapus collection docs sebelum insert",
)

args = parser.parse_args()

# ============================================================
# VALIDASI MODE
# ============================================================

if args.all and args.scenario:
    parser.error("Gunakan --all ATAU --scenario, jangan keduanya.")

if args.input and args.dir:
    parser.error("--input dan --dir tidak boleh digunakan bersamaan.")

if args.input and args.files:
    parser.error("--input dan --files tidak boleh digunakan bersamaan.")

if args.top_k is not None and args.top_k <= 0:
    parser.error("top_k harus lebih besar dari 0.")

if args.chunk_size is not None and args.chunk_size <= 0:
    parser.error("chunk_size harus lebih besar dari 0.")

if args.overlap is not None and args.overlap < 0:
    parser.error("overlap tidak boleh negatif.")

# ============================================================
# EMBEDDER
# ============================================================

print("\nMemuat embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("Embedding model siap.")

# ============================================================
# HELPER
# ============================================================

def make_unique_id(source_file, chunk_id, index):
    """
    Membuat ID global yang unik.
    Penting karena setiap PDF sekarang memiliki JSON terpisah.
    """

    raw = f"{Path(source_file).name}|{chunk_id}|{index}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def make_text(item):
    """
    Membentuk teks yang akan di-embedding.
    """

    return (
        f"{item.get('title', '')}\n"
        f"Kategori: {item.get('category', '')}\n"
        f"Sub: {item.get('sub_category', '')}\n"
        f"Isi: {item.get('content', '')}"
    ).strip()


def make_metadata(item, source_file, scenario_id=None):
    """
    Membentuk metadata ChromaDB.
    """

    keywords = item.get("keywords", [])

    if isinstance(keywords, list):
        keywords = ", ".join(
            str(x) for x in keywords
        )

    metadata = {
        "title": str(
            item.get("title", "")
        ),
        "category": str(
            item.get("category", "")
        ),
        "sub_category": str(
            item.get("sub_category", "")
        ),
        "type": str(
            item.get("type", "")
        ),
        "keywords": str(keywords),
        "priority": str(
            item.get("priority", "")
        ),
        "page": str(
            item.get(
                "page",
                item.get("start_page", "")
            )
        ),
        "source": str(
            item.get("source", "")
        ),
        "source_file": Path(
            source_file
        ).name,
    }

    if scenario_id:
        metadata["scenario"] = scenario_id

    return metadata


def upsert_batches(
    collection,
    documents,
    embeddings,
    ids,
    metadatas,
):
    """
    Insert/upsert ke ChromaDB secara bertahap.
    Ini mencegah error:
    Batch size ... greater than max batch size ...
    """

    total = len(documents)

    for start in range(
        0,
        total,
        CHROMA_BATCH_SIZE,
    ):
        end = min(
            start + CHROMA_BATCH_SIZE,
            total,
        )

        collection.upsert(
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )

        print(
            f"  [UPSERT] {end}/{total} chunks"
        )

# ============================================================
# FUNGSI EMBEDDING
# ============================================================

def embed_file(
    file_path,
    chroma_path,
    top_k,
    chunk_size,
    overlap,
    reset=False,
    scenario_id=None,
):
    """
    Memasukkan satu file JSON chunk ke satu ChromaDB.

    Satu file JSON = satu PDF hasil chunking.
    Banyak file JSON dapat masuk ke collection docs
    yang sama dalam satu skenario.
    """

    file_path = Path(file_path)
    chroma_path = Path(chroma_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"File chunk tidak ditemukan:\n{file_path}"
        )

    chroma_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("EMBEDDING")
    print("=" * 70)
    print(f"Input      : {file_path}")
    print(f"Chroma     : {chroma_path}")
    print(f"Top-K      : {top_k}")
    print(f"Chunk Size : {chunk_size}")
    print(f"Overlap    : {overlap}")
    print("=" * 70)

    client = chromadb.PersistentClient(
        path=str(chroma_path)
    )

    if reset:
        try:
            client.delete_collection("docs")
            print(
                "[OK] Collection lama dihapus."
            )
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name="docs"
    )

    try:
        with open(
            file_path,
            "r",
            encoding="utf-8",
        ) as f:
            data = json.load(f)

    except Exception as e:
        raise RuntimeError(
            f"Gagal membaca {file_path}: {e}"
        )

    if not isinstance(data, list):
        raise RuntimeError(
            f"Format JSON harus berupa list: {file_path}"
        )

    documents = []
    ids = []
    metadatas = []

    seen_ids = set()

    # --------------------------------------------------------
    # PREPARE TEXT
    # --------------------------------------------------------

    texts_to_embed = []
    pending_ids = []
    pending_metadatas = []

    for i, item in enumerate(data):

        try:

            if not isinstance(item, dict):
                continue

            chunk_id_raw = item.get(
                "chunk_id"
            )

            if chunk_id_raw:
                base_chunk_id = str(
                    chunk_id_raw
                )
            else:
                base_chunk_id = hashlib.md5(
                    (
                        str(
                            item.get(
                                "source",
                                ""
                            )
                        )
                        + "|"
                        + str(
                            item.get(
                                "content",
                                ""
                            )
                        )[:200]
                    ).encode("utf-8")
                ).hexdigest()

            chunk_id = make_unique_id(
                file_path,
                base_chunk_id,
                i,
            )

            if chunk_id in seen_ids:
                print(
                    f"[SKIP] duplicate: {chunk_id}"
                )
                continue

            text = make_text(item)

            if not text.strip():
                print(
                    f"[SKIP] chunk kosong ke-{i}"
                )
                continue

            seen_ids.add(chunk_id)

            texts_to_embed.append(text)
            pending_ids.append(chunk_id)
            pending_metadatas.append(
                make_metadata(
                    item,
                    file_path,
                    scenario_id,
                )
            )

        except Exception as e:

            print(
                f"[ERROR] item ke-{i}: {e}"
            )

    # --------------------------------------------------------
    # EMBEDDING BATCH
    # --------------------------------------------------------

    total_items = len(texts_to_embed)

    for start in range(
        0,
        total_items,
        EMBED_BATCH_SIZE,
    ):

        end = min(
            start + EMBED_BATCH_SIZE,
            total_items,
        )

        batch_texts = texts_to_embed[
            start:end
        ]

        batch_embeddings = embedder.encode(
            batch_texts,
            batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=False,
        ).tolist()

        documents.extend(
            batch_texts
        )

        ids.extend(
            pending_ids[start:end]
        )

        metadatas.extend(
            pending_metadatas[start:end]
        )

        print(
            f"  [EMBED] {end}/{total_items}"
        )

        # Simpan embedding batch langsung
        # agar tidak menunggu ribuan chunk.
        upsert_batches(
            collection=collection,
            documents=batch_texts,
            embeddings=batch_embeddings,
            ids=pending_ids[start:end],
            metadatas=pending_metadatas[start:end],
        )

    if documents:
        print(
            f"[OK] {len(documents)} chunk "
            f"dari {file_path.name} masuk ke Chroma."
        )
    else:
        print(
            "[WARNING] Tidak ada chunk "
            "yang berhasil diinsert."
        )

    return len(documents)

# ============================================================
# RESOLVE FILE
# ============================================================

def resolve_files():
    """
    Menentukan file JSON yang akan diproses
    jika menggunakan mode manual.
    """

    default_dir = (
        PROJECT_ROOT
        / "data"
        / "chunks"
    )

    if args.input:
        return [Path(args.input)]

    if args.files:
        return [
            Path(x)
            for x in args.files
        ]

    if args.dir:

        directory = Path(
            args.dir
        )

        return sorted(
            directory.glob("*.json")
        )

    return sorted(
        default_dir.glob("*.json")
    )

# ============================================================
# SINGLE SCENARIO
# ============================================================

def run_single_scenario(
    scenario_id
):
    """
    Embed semua JSON dalam:
    data/chunks/Sx/

    menjadi satu vectorstore:
    vectorstore/Sx/
    """

    config = SCENARIOS[
        scenario_id
    ]

    scenario_dir = (
        PROJECT_ROOT
        / "data"
        / "chunks"
        / scenario_id
    )

    chroma_path = (
        PROJECT_ROOT
        / "vectorstore"
        / scenario_id
    )

    json_files = sorted(
        scenario_dir.glob(
            "*.json"
        )
    )

    if not json_files:
        raise FileNotFoundError(
            "Tidak ditemukan file JSON "
            f"di:\n{scenario_dir}"
        )

    print()
    print("#" * 70)
    print(
        f"# EMBEDDING SKENARIO {scenario_id}"
    )
    print("#" * 70)
    print(
        f"Folder chunk : {scenario_dir}"
    )
    print(
        f"Jumlah JSON  : {len(json_files)}"
    )
    print(
        f"Vector DB    : {chroma_path}"
    )
    print(
        f"Chunk Size   : {config['chunk_size']}"
    )
    print(
        f"Overlap      : {config['overlap']}"
    )
    print(
        f"Top-K        : {config['top_k']}"
    )
    print("#" * 70)

    # Hapus collection sekali saja.
    client = chromadb.PersistentClient(
        path=str(chroma_path)
    )

    try:
        client.delete_collection(
            "docs"
        )
        print(
            "[OK] Collection lama dihapus."
        )
    except Exception:
        pass

    total = 0
    file_results = []

    for index, chunk_file in enumerate(
        json_files,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(json_files)}] "
            f"Embedding: {chunk_file.name}"
        )

        count = embed_file(
            file_path=chunk_file,
            chroma_path=chroma_path,
            top_k=config["top_k"],
            chunk_size=config["chunk_size"],
            overlap=config["overlap"],
            reset=False,
            scenario_id=scenario_id,
        )

        total += count

        file_results.append(
            {
                "file": str(
                    chunk_file
                ),
                "document_count": count,
            }
        )

    return {
        "scenario": scenario_id,
        "chunk_size": config[
            "chunk_size"
        ],
        "overlap": config[
            "overlap"
        ],
        "top_k": config["top_k"],
        "input_dir": str(
            scenario_dir
        ),
        "input_json_count": len(
            json_files
        ),
        "chroma_path": str(
            chroma_path
        ),
        "document_count": total,
        "files": file_results,
    }

# ============================================================
# ALL SCENARIOS
# ============================================================

def run_all_scenarios():

    results = {}

    for scenario_id in SCENARIOS:

        results[scenario_id] = (
            run_single_scenario(
                scenario_id
            )
        )

    return results

# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # SINGLE SCENARIO
    # --------------------------------------------------------

    if args.scenario:

        result = run_single_scenario(
            args.scenario
        )

        print()
        print(
            "[OK] Skenario selesai."
        )

        print(
            json.dumps(
                result,
                ensure_ascii=False,
                indent=2,
            )
        )

        return

    # --------------------------------------------------------
    # ALL
    # --------------------------------------------------------

    if args.all:

        results = run_all_scenarios()

        summary_path = (
            PROJECT_ROOT
            / "results"
            / "embedding_summary.json"
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
        print("#" * 70)
        print(
            "# SEMUA EMBEDDING SELESAI"
        )
        print("#" * 70)

        for (
            scenario_id,
            result,
        ) in results.items():

            print(
                f"{scenario_id}: "
                f"{result['input_json_count']} JSON | "
                f"{result['document_count']} chunks "
                f"-> {result['chroma_path']}"
            )

        print(
            f"\n[OK] Summary: {summary_path}"
        )

        return

    # --------------------------------------------------------
    # MANUAL MODE
    # --------------------------------------------------------

    files = resolve_files()

    if not files:
        parser.error(
            "Tidak ada file JSON untuk diproses."
        )

    top_k = (
        args.top_k
        if args.top_k is not None
        else 5
    )

    chunk_size = (
        args.chunk_size
        if args.chunk_size is not None
        else 500
    )

    overlap = (
        args.overlap
        if args.overlap is not None
        else 50
    )

    chroma_path = (
        Path(args.chroma_path)
        if args.chroma_path
        else (
            PROJECT_ROOT
            / "vectorstore"
            / "default"
        )
    )

    total = 0

    for index, file_path in enumerate(
        files
    ):

        total += embed_file(
            file_path=file_path,
            chroma_path=chroma_path,
            top_k=top_k,
            chunk_size=chunk_size,
            overlap=overlap,
            reset=(
                args.reset
                and index == 0
            ),
        )

    print(
        f"\n[OK] Total {total} chunk "
        "masuk ke Chroma."
    )

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
