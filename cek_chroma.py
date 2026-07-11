import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="docs")

#  total data
total = collection.count()
print("TOTAL DATA:", total)

# ambil contoh 5 data
data = collection.get(limit=5)

print("\nIDS:")
print(data["ids"])

print("\nDOCUMENTS:")
for doc in data["documents"]:
    print(doc[:200], "\n")

print("\nMETADATA:")
for meta in data["metadatas"]:
    print(meta)