"""
Avaliação RAG — Secretária Virtual
===================================
Métricas implementadas (equivalentes às do framework RAGAS em modo embedding-only):

  1. Answer Similarity   — cosine similarity entre embeddings da resposta gerada e ground truth
  2. Context Precision   — cosine similarity média entre pergunta e contextos recuperados
  3. Context Recall      — cosine similarity máxima entre ground truth e contextos recuperados
  4. Faithfulness Proxy  — cosine similarity média entre resposta gerada e contextos recuperados
  5. BERTScore F1        — similaridade semântica token-level com bert-base-multilingual-cased
  6. ROUGE-L             — sobreposição de subsequência mais longa (metric lexical clássica)

As métricas 1-4 são calculadas com o mesmo modelo de embedding (all-MiniLM-L6-v2)
que o pipeline RAG usa internamente — garantindo consistência com o vectorstore.
"""

import os
import ast
import json
import random
import pickle
import argparse
import datetime

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
LORA_PATH = "lora_model"
VECTORSTORE_DIR = "vectorstore"
DATASET_PATH = "data/dataset_secretaria_virtual_5000.json"
OUTPUT_DIR = "evaluation_results"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def carregar_dataset(caminho):
    amostras = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            try:
                amostras.append(ast.literal_eval(linha))
            except Exception:
                amostras.append(json.loads(linha))
    return amostras


def criar_conjunto_teste_estratificado(amostras, n_por_intencao=17, seed=42):
    """~17 amostras por intenção, seed fixo para reprodutibilidade."""
    random.seed(seed)
    por_intencao = {}
    for amostra in amostras:
        chave = amostra.get("instruction", "desconhecido")
        por_intencao.setdefault(chave, []).append(amostra)

    conjunto = []
    for grupo in por_intencao.values():
        conjunto.extend(random.sample(grupo, min(n_por_intencao, len(grupo))))

    random.shuffle(conjunto)
    return conjunto


# ---------------------------------------------------------------------------
# Componentes RAG
# ---------------------------------------------------------------------------

def carregar_componentes():
    print("[1/5] Configurando quantização 4-bit NF4...")
    # llm_int8_enable_fp32_cpu_offload é exclusivo do modo 8-bit e causa
    # KeyError 'lm_head' no PEFT 0.19 quando combinado com device_map="auto".
    # Para 4-bit (nf4), device_map="auto" já gerencia a memória entre GPU e CPU.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    print("[2/5] Carregando tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

    print("[3/5] Carregando Mistral-7B (4-bit, device_map auto)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    print("[4/5] Aplicando LoRA fine-tuning...")
    model = PeftModel.from_pretrained(base_model, LORA_PATH)

    print("[5/5] Carregando embedding model e vectorstore FAISS...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index(f"{VECTORSTORE_DIR}/faiss.index")
    with open(f"{VECTORSTORE_DIR}/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    print("Componentes carregados.\n")
    return tokenizer, model, embedding_model, index, texts


# ---------------------------------------------------------------------------
# Pipeline RAG
# ---------------------------------------------------------------------------

def recuperar_contextos(pergunta, embedding_model, index, texts, top_k=3):
    """Retorna List[str] dos top-k textos mais similares (RAGAS exige lista)."""
    emb = embedding_model.encode([pergunta])
    _, indices = index.search(emb, top_k)
    return [texts[idx] for idx in indices[0]]


def gerar_resposta(pergunta, contextos_lista, tokenizer, model):
    """Mesmo prompt template de rag_chat.py; memória vazia (avaliação single-turn)."""
    data_atual = datetime.datetime.now().strftime("%d/%m/%Y")
    contexto_str = "\n\n".join(contextos_lista)

    prompt = f"""<s>[INST] Você é a secretária de uma clínica médica. Hoje é {data_atual}.

Sua missão é coletar 4 dados para agendar a consulta: [Data], [Horário], [Nome do Paciente] e [Plano de Saúde].

REGRAS OBRIGATÓRIAS:
1. LEIA O HISTÓRICO: Se o paciente já informou a Data (ex: 13/05), NÃO PERGUNTE A DATA NOVAMENTE. Passe para o próximo dado que falta (ex: Horário).
2. NUNCA faça a mesma pergunta duas vezes. Se a resposta já está no texto, aceite-a.
3. CONFIRMAÇÃO: Quando tiver os 4 dados preenchidos, resuma e pergunte: "Posso confirmar o agendamento?".
4. ENCERRAMENTO: Se o paciente disser "Sim" para a confirmação final, responda EXATAMENTE E APENAS: [FINALIZADO].

[Base de Conhecimento]
{contexto_str}

[Histórico da Conversa]


[Mensagem Atual]
Paciente disse: {pergunta}

Escreva sua resposta curta, direta e educada para a Mensagem Atual: [/INST]"""

    # Com device_map="auto" o modelo pode estar dividido entre GPU e CPU.
    # Usamos o device do embedding (primeira camada), que é sempre GPU quando disponível.
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.3,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    texto = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Libera a VRAM após cada geração para evitar acúmulo nas 102 iterações
    del inputs, outputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return texto.split("[/INST]")[-1].strip()


# ---------------------------------------------------------------------------
# Coleta de resultados
# ---------------------------------------------------------------------------

def coletar_resultados(amostras_teste, tokenizer, model, embedding_model, index, texts):
    dados = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for amostra in tqdm(amostras_teste, desc="Inferência RAG"):
        pergunta = amostra["input"]
        ground_truth = amostra["output"]
        contextos = recuperar_contextos(pergunta, embedding_model, index, texts, top_k=3)
        try:
            resposta = gerar_resposta(pergunta, contextos, tokenizer, model)
        except Exception as e:
            print(f"\nAVISO: erro na amostra '{pergunta[:50]}': {e}")
            resposta = ""  # mantém a amostra com resposta vazia em vez de abortar

        dados["question"].append(pergunta)
        dados["answer"].append(resposta)
        dados["contexts"].append(contextos)
        dados["ground_truth"].append(ground_truth)

    return dados


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _embed(textos, modelo):
    return modelo.encode(textos, normalize_embeddings=True)


def calcular_answer_similarity(respostas, ground_truths, embedding_model):
    """
    Cosine similarity entre embedding da resposta e do ground truth.
    Equivalente a RAGAS answer_similarity em modo embedding-only.
    """
    emb_r = _embed(respostas, embedding_model)
    emb_g = _embed(ground_truths, embedding_model)
    sims = [float(cosine_similarity([r], [g])[0][0]) for r, g in zip(emb_r, emb_g)]
    return sims


def calcular_context_precision(perguntas, contextos_lista, embedding_model):
    """
    Média da cosine similarity entre pergunta e cada contexto recuperado.
    Equivalente a RAGAS context_precision em modo embedding-only.
    """
    scores = []
    for pergunta, contextos in zip(perguntas, contextos_lista):
        emb_p = _embed([pergunta], embedding_model)
        emb_c = _embed(contextos, embedding_model)
        sims = cosine_similarity(emb_p, emb_c)[0]
        scores.append(float(sims.mean()))
    return scores


def calcular_context_recall(ground_truths, contextos_lista, embedding_model):
    """
    Máximo da cosine similarity entre ground truth e contextos recuperados.
    Mede se ao menos um contexto cobre a resposta esperada.
    Equivalente a RAGAS context_recall em modo embedding-only.
    """
    scores = []
    for gt, contextos in zip(ground_truths, contextos_lista):
        emb_gt = _embed([gt], embedding_model)
        emb_c = _embed(contextos, embedding_model)
        sims = cosine_similarity(emb_gt, emb_c)[0]
        scores.append(float(sims.max()))
    return scores


def calcular_faithfulness_proxy(respostas, contextos_lista, embedding_model):
    """
    Média da cosine similarity entre resposta gerada e contextos recuperados.
    Proxy para faithfulness: resposta fiel ao contexto deve ser semanticamente próxima.
    """
    scores = []
    for resposta, contextos in zip(respostas, contextos_lista):
        emb_r = _embed([resposta], embedding_model)
        emb_c = _embed(contextos, embedding_model)
        sims = cosine_similarity(emb_r, emb_c)[0]
        scores.append(float(sims.mean()))
    return scores


def calcular_bert_score(respostas, ground_truths):
    """BERTScore com bert-base-multilingual-cased (suporte nativo a PT-BR)."""
    from bert_score import score as bs_score
    P, R, F1 = bs_score(respostas, ground_truths, lang="pt", verbose=False)
    return (
        [round(p.item(), 4) for p in P],
        [round(r.item(), 4) for r in R],
        [round(f.item(), 4) for f in F1],
    )


def calcular_rouge_l(respostas, ground_truths):
    """ROUGE-L F1 — captura sobreposição de subsequência mais longa."""
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    scores = []
    for r, g in zip(respostas, ground_truths):
        resultado = scorer.score(g, r)
        scores.append(round(resultado["rougeL"].fmeasure, 4))
    return scores


# ---------------------------------------------------------------------------
# Relatório
# ---------------------------------------------------------------------------

def imprimir_relatorio(df):
    n = len(df)
    print("\n" + "=" * 64)
    print("   RELATÓRIO DE AVALIAÇÃO RAG — SECRETÁRIA VIRTUAL")
    print("=" * 64)
    print(f"  Amostras avaliadas : {n}")
    print(f"\n  NOTA: As {n} amostras pertencem ao vectorstore de")
    print(f"  treinamento. Métricas de contexto tendem a ser elevadas.")
    print(f"  Para avaliação real, use dados nunca vistos pelo modelo.\n")

    grupos = [
        ("Métricas de Resposta (embedding all-MiniLM-L6-v2)",
         ["answer_similarity", "faithfulness_proxy"]),
        ("Métricas de Recuperação (FAISS)",
         ["context_precision", "context_recall"]),
        ("BERTScore (bert-base-multilingual-cased)",
         ["bertscore_precision", "bertscore_recall", "bertscore_f1"]),
        ("Métrica Lexical",
         ["rouge_l"]),
    ]
    for titulo, colunas in grupos:
        print(f"  --- {titulo} ---")
        for col in colunas:
            if col in df.columns:
                print(f"  {col:<28} {df[col].mean():.4f}")
        print()

    metricas_ragas = ["answer_similarity", "context_precision", "context_recall", "faithfulness_proxy"]
    vals = [df[m].mean() for m in metricas_ragas if m in df.columns]
    if vals:
        # Harmonic mean das 4 métricas — análogo ao RAGAS score
        ragas_score = len(vals) / sum(1 / v for v in vals if v > 0)
        print(f"  {'RAGAS Score (média harmônica)':<28} {ragas_score:.4f}")
    print("=" * 64 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Avaliação RAGAS-equivalente — Secretária Virtual")
    parser.add_argument("--max_amostras", type=int, default=None,
                        help="Limita número de amostras (ex: 10 para teste rápido)")
    args = parser.parse_args()

    print(">> Carregando dataset...")
    amostras = carregar_dataset(DATASET_PATH)
    conjunto_teste = criar_conjunto_teste_estratificado(amostras, n_por_intencao=17, seed=42)

    if args.max_amostras:
        conjunto_teste = conjunto_teste[: args.max_amostras]

    print(f">> Conjunto de teste estratificado: {len(conjunto_teste)} amostras\n")

    tokenizer, model, embedding_model, index, texts = carregar_componentes()

    print(">> Executando pipeline RAG...")
    dados = coletar_resultados(conjunto_teste, tokenizer, model, embedding_model, index, texts)

    perguntas    = dados["question"]
    respostas    = dados["answer"]
    contextos    = dados["contexts"]
    ground_truths = dados["ground_truth"]

    print("\n>> Calculando métricas de embedding...")
    sim        = calcular_answer_similarity(respostas, ground_truths, embedding_model)
    cp         = calcular_context_precision(perguntas, contextos, embedding_model)
    cr         = calcular_context_recall(ground_truths, contextos, embedding_model)
    faith      = calcular_faithfulness_proxy(respostas, contextos, embedding_model)

    print(">> Calculando BERTScore...")
    bs_p, bs_r, bs_f1 = calcular_bert_score(respostas, ground_truths)

    print(">> Calculando ROUGE-L...")
    rouge_l    = calcular_rouge_l(respostas, ground_truths)

    df = pd.DataFrame({
        "question":           perguntas,
        "ground_truth":       ground_truths,
        "answer":             respostas,
        "answer_similarity":  sim,
        "context_precision":  cp,
        "context_recall":     cr,
        "faithfulness_proxy": faith,
        "bertscore_precision": bs_p,
        "bertscore_recall":   bs_r,
        "bertscore_f1":       bs_f1,
        "rouge_l":            rouge_l,
    })

    imprimir_relatorio(df)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(f"{OUTPUT_DIR}/ragas_results.csv", index=False, encoding="utf-8-sig")

    summary = {
        "n_amostras": len(df),
        "metricas": {
            "answer_similarity":   round(df["answer_similarity"].mean(), 4),
            "context_precision":   round(df["context_precision"].mean(), 4),
            "context_recall":      round(df["context_recall"].mean(), 4),
            "faithfulness_proxy":  round(df["faithfulness_proxy"].mean(), 4),
            "bertscore_precision": round(df["bertscore_precision"].mean(), 4),
            "bertscore_recall":    round(df["bertscore_recall"].mean(), 4),
            "bertscore_f1":        round(df["bertscore_f1"].mean(), 4),
            "rouge_l":             round(df["rouge_l"].mean(), 4),
        }
    }
    ragas_vals = [
        summary["metricas"]["answer_similarity"],
        summary["metricas"]["context_precision"],
        summary["metricas"]["context_recall"],
        summary["metricas"]["faithfulness_proxy"],
    ]
    summary["ragas_score"] = round(4 / sum(1 / v for v in ragas_vals if v > 0), 4)

    with open(f"{OUTPUT_DIR}/ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Resultados salvos em '{OUTPUT_DIR}/'")
    print(f"  - {OUTPUT_DIR}/ragas_results.csv   (por amostra)")
    print(f"  - {OUTPUT_DIR}/ragas_summary.json  (resumo agregado)")


if __name__ == "__main__":
    main()
