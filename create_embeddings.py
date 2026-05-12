import os
import ast
import pickle
import faiss

from sentence_transformers import SentenceTransformer

# ==========================================
# MODELO EMBEDDING
# ==========================================

print("Carregando modelo embedding...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

print("Modelo carregado!")

# ==========================================
# DATASET
# ==========================================

dataset_path = "data/dataset_secretaria_virtual_5000.json"

texts = []

print("Lendo dataset...")

with open(dataset_path, encoding="utf-8") as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        item = ast.literal_eval(line)

        pergunta = item["input"]
        resposta = item["output"]

        texto = f"""
Pergunta:
{pergunta}

Resposta:
{resposta}
"""

        texts.append(texto)

print(f"Total de textos: {len(texts)}")

# ==========================================
# EMBEDDINGS
# ==========================================

print("Gerando embeddings...")

embeddings = embedding_model.encode(
    texts,
    show_progress_bar=True
)

print("Embeddings gerados!")

# ==========================================
# FAISS
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("FAISS criado!")

# ==========================================
# SALVAR
# ==========================================

os.makedirs("vectorstore", exist_ok=True)

faiss.write_index(
    index,
    "vectorstore/faiss.index"
)

with open(
    "vectorstore/texts.pkl",
    "wb"
) as f:

    pickle.dump(texts, f)

print("Vectorstore salvo!")