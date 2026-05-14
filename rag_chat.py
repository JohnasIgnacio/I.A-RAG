import datetime
import os
# Impede conflitos do OpenMP no Windows
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

print(">> PASSO 0: Iniciando script...")

import pickle
print(">> PASSO 1: Pickle OK!")

import sklearn
print(">> PASSO 2: Sklearn OK!")

from sentence_transformers import SentenceTransformer
print(">> PASSO 3: SentenceTransformers OK!")

import torch
print(">> PASSO 4: Torch OK!")

import faiss
print(">> PASSO 5: Faiss OK!")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
print(">> PASSO 6: Transformers OK!")

from peft import PeftModel
print(">> PASSO 7: Peft OK!")

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print(">> PASSO 8: Configurando BitsAndBytes com Offload para CPU...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True  # A SOLUÇÃO: Permite usar a RAM do PC para o que não couber na GPU
)

print("\n--- Todas as bibliotecas passaram! Iniciando Secretária Virtual ---\n")

try:
    print("[1/5] Carregando Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    print("[2/5] Carregando Mistral em 4-bits (Dividindo entre GPU e CPU, aguarde)...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto" # Com o Offload ativado, o "auto" agora gerencia a divisão de memória perfeitamente
    )

    print("[3/5] Aplicando inteligência da Secretária (LoRA)...")
    model = PeftModel.from_pretrained(base_model, "lora_model")
    
    print("[4/5] Carregando base de conhecimento (RAG)...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index = faiss.read_index("vectorstore/faiss.index")
    with open("vectorstore/texts.pkl", "rb") as f:
        texts = pickle.load(f)

    print("[5/5] Sistema pronto!")
    print("\n" + "="*40)
    print(" SECRETÁRIA ONLINE - Pressione Ctrl+C para sair")
    print("="*40 + "\n")

except Exception as e:
    print(f"\nERRO AO INICIAR: {e}")
    exit()

# ==========================================
# FUNÇÃO DE BUSCA E CHAT COM MEMÓRIA
# ==========================================
def buscar_contexto(pergunta, top_k=3):
    pergunta_embedding = embedding_model.encode([pergunta])
    distancias, indices = index.search(pergunta_embedding, top_k)
    return "\n\n".join([texts[idx] for idx in indices[0]])

# Essa lista vai guardar as últimas mensagens da conversa
historico_chat = []

while True:
    pergunta = input("Você: ")
    if pergunta.lower() in ["sair", "exit", "quit"]:
        break

    contexto = buscar_contexto(pergunta)

  
    memoria_str = ""
    for turno in historico_chat[-4:]:
        memoria_str += f"Paciente disse: {turno['user']}\nVocê perguntou: {turno['bot']}\n"

    # Pega a data atual
    import datetime
    data_atual = datetime.datetime.now().strftime("%d/%m/%Y")

    # ==========================================
    # SUPER PROMPT: O CHECKLIST DE ATENDIMENTO
    # ==========================================
    # ==========================================
    # SUPER PROMPT v2.0: O CHECKLIST DE FERRO
    # ==========================================
    # ==========================================
    # SUPER PROMPT v2.0: O CHECKLIST DE FERRO
    # ==========================================
    # ==========================================
    # SUPER PROMPT v3.0: BLOCOS DE ATENÇÃO
    # ==========================================
    prompt = f"""<s>[INST] Você é a secretária de uma clínica médica. Hoje é {data_atual}.

Sua missão é coletar 4 dados para agendar a consulta: [Data], [Horário], [Nome do Paciente] e [Plano de Saúde].

REGRAS OBRIGATÓRIAS:
1. LEIA O HISTÓRICO: Se o paciente já informou a Data (ex: 13/05), NÃO PERGUNTE A DATA NOVAMENTE. Passe para o próximo dado que falta (ex: Horário).
2. NUNCA faça a mesma pergunta duas vezes. Se a resposta já está no texto, aceite-a.
3. CONFIRMAÇÃO: Quando tiver os 4 dados preenchidos, resuma e pergunte: "Posso confirmar o agendamento?".
4. ENCERRAMENTO: Se o paciente disser "Sim" para a confirmação final, responda EXATAMENTE E APENAS: [FINALIZADO].

[Base de Conhecimento]
{contexto}

[Histórico da Conversa]
{memoria_str}

[Mensagem Atual]
Paciente disse: {pergunta}

Escreva sua resposta curta, direta e educada para a Mensagem Atual: [/INST]"""

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150,
            temperature=0.3, 
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )

    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    resposta_final = resposta.split("[/INST]")[-1].strip()

    # ==========================================
    # O TRUQUE DE REINICIAR O CHAT
    # ==========================================
    if "[FINALIZADO]" in resposta_final:
        # Tira a palavra feia da frente do paciente
        resposta_final = resposta_final.replace("[FINALIZADO]", "").strip()
        print(f"\nSecretária: {resposta_final}\n")
        print("=" * 40)
        print(" SESSÃO FINALIZADA - A MEMÓRIA DA SECRETÁRIA FOI ZERADA ")
        print("=" * 40 + "\n")
        
        # Limpa o histórico para o próximo paciente!
        historico_chat = []
        continue # Pula a linha de baixo e volta pro começo do 'while'

    # Se não finalizou, salva a conversa normalmente
    historico_chat.append({"user": pergunta, "bot": resposta_final})

    print(f"\nSecretária: {resposta_final}\n")
    print("-" * 40)