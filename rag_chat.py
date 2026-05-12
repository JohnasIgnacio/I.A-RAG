import pickle
import faiss
import torch

from sentence_transformers import SentenceTransformer

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

# ==========================================
# MISTRAL
# ==========================================

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Carregando Mistral...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Mistral carregado!")

# ==========================================
# EMBEDDINGS
# ==========================================

print("Carregando embeddings...")

embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# ==========================================
# FAISS
# ==========================================

print("Carregando FAISS...")

index = faiss.read_index(
    "vectorstore/faiss.index"
)

with open(
    "vectorstore/texts.pkl",
    "rb"
) as f:

    texts = pickle.load(f)

print("FAISS carregado!")

# ==========================================
# BUSCA CONTEXTO
# ==========================================

def buscar_contexto(pergunta, top_k=3):

    pergunta_embedding = embedding_model.encode(
        [pergunta]
    )

    distancias, indices = index.search(
        pergunta_embedding,
        top_k
    )

    contextos = []

    for idx in indices[0]:

        contextos.append(texts[idx])

    return "\n\n".join(contextos)

# ==========================================
# CHAT
# ==========================================

print("\nRAG iniciado!")
print("Digite 'sair' para encerrar.\n")

while True:

    pergunta = input("Você: ")

    if pergunta.lower() == "sair":
        break

    contexto = buscar_contexto(pergunta)

    prompt = f"""
<s>[INST]

Você é uma secretária virtual.

Use o contexto abaixo para responder.

Contexto:
{contexto}

Pergunta:
{pergunta}

[/INST]
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=100
        )

    resposta = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print("\nIA:")
    print(resposta)
    print("\n")