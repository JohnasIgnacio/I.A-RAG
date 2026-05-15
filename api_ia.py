import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sklearn
import re

from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI()

# --- CARREGAMENTO DO MODELO (Roda só uma vez ao ligar a API) ---
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)

print("Carregando IA, aguarde...")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto")
print("IA PRONTA PARA RECEBER REQUISIÇÕES!")

# --- ESTRUTURA DE DADOS RECEBIDA DO JAVA ---
class RequisicaoChat(BaseModel):
    pergunta: str
    historico: str       # O Java vai mandar as últimas mensagens
    contexto_banco: str  # O Java vai mandar os dados reais do banco (ex: horários livres)

# --- ROTA DA API ---
@app.post("/api/gerar-resposta")
def gerar_resposta(req: RequisicaoChat):
    max_tentativas = 3
    melhor_resposta = ""

    for tentativa in range(max_tentativas):
        print(f"\n--- TENTATIVA {tentativa + 1} de {max_tentativas} ---")
        # ==========================================
        # PASSO 1: GERAÇÃO DA RESPOSTA CANDIDATA
        # ==========================================
        prompt_geracao = f"""<s>[INST] Você é o Assistente Interno da clínica.
Sua função é ler a mensagem do administrador, coletar os dados e, ao final, gerar o comando para agendar no Google Agenda.

CHECKLIST OBRIGATÓRIO:
1. CPF (Usado para validar e buscar o paciente)
2. Data da consulta (formato DD/MM)
3. Horário
4. Email do paciente
5. Descrição (motivo da consulta)
6. Confirmação Final (Resumo de todos os dados e aprovação do paciente)

REGRAS DE FUNCIONAMENTO:
1. A primeira informação a pedir é SEMPRE o CPF. Leia o [HISTÓRICO DA CONVERSA] para verificar se o paciente já o forneceu. NUNCA peça o CPF se já estiver no histórico.
2. AO RECEBER O CPF: Verifique no bloco [DADOS DO BANCO]. 
   - Se o CPF NÃO estiver no banco, responda APENAS: "Paciente não encontrado."
   - Se o CPF ESTIVER no banco, PROIBIDO dizer que "vai verificar". Diga que encontrou o paciente e IMEDIATAMENTE faça uma pergunta solicitando os dados que ainda faltam do checklist.
3. MEMÓRIA: Leia o [HISTÓRICO DA CONVERSA]. Se a data, horário, email ou descrição já foram passados, não pergunte novamente.
4. ETAPA DE CONFIRMAÇÃO: Quando tiver coletado os 5 primeiros itens, faça um RESUMO CLARO com todos os dados e pergunte: "Posso confirmar o agendamento?".
5. AÇÃO FINAL: APENAS DEPOIS que o paciente responder "sim" à sua confirmação, gere o comando de agendamento.
6. PÓS-AGENDAMENTO: Se a tag [GERAR_AGENDA_GOOGLE] já estiver presente no [HISTÓRICO DA CONVERSA], significa que o atendimento já acabou. Responda APENAS: "O agendamento já foi processado no sistema." e NUNCA reinicie o checklist.

FORMATO OBRIGATÓRIO DA AÇÃO FINAL (Sem tabelas, em uma única linha):
[GERAR_AGENDA_GOOGLE] Nome | CPF | Data | Horário | Email | Descrição
(Atenção: NÃO use chaves como "{{data}}" ou caracteres de escape na tag. Use APENAS os dados reais).

REGRAS DE SEGURANÇA MÁXIMA:
1. NUNCA invente ou adivinhe dados.
2. É ESTRITAMENTE PROIBIDO usar a tag [GERAR_AGENDA_GOOGLE] se faltar qualquer dado ou ANTES do paciente confirmar o resumo.
3. PROIBIDO fazer perguntas de confirmação no MEIO do processo (ex: "O CPF é válido?", "Podemos definir a data?"). A ÚNICA confirmação permitida é o resumo final. Peça DIRETAMENTE o que falta.
4. PROIBIDO repetir perguntas. Se o usuário acabou de digitar a data (ex: 13/05), NÃO PERGUNTE de novo. Passe direto para o próximo dado faltante.
5. IDIOMA E SIGILO: Responda EXCLUSIVAMENTE em Português do Brasil (PT-BR). É terminantemente PROIBIDO falar inglês ou gerar a tag [DADOS DO BANCO] na sua resposta. Encerre sua fala assim que fizer a pergunta.
6. ESTILO DE RESPOSTA: Fale diretamente como um atendente humano. Faça APENAS UMA pergunta por vez e PARE DE ESCREVER. NUNCA escreva regras de sistema, opções, condicionais (ex: "Se o paciente...") ou roteiros. É PROIBIDO gerar a tag [GERAR_AGENDA_GOOGLE] na mesma resposta em que faz uma pergunta.


[DADOS DO BANCO]
{req.contexto_banco}

[HISTÓRICO DA CONVERSA]
{req.historico}

Mensagem atual do administrador: {req.pergunta}
Sua resposta: [/INST]"""
        
        inputs = tokenizer(prompt_geracao, return_tensors="pt").to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=150,
                temperature=0.4, # Aumentado para permitir variação se ela errar
                do_sample=True,          
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id
            )

        resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
        resposta_candidata = resposta.split("[/INST]")[-1].strip()
        
        # Filtro antialucinação agressivo: corta a string se a IA começar a narrar regras ou vazar prompt
        cortes = [
            r"\(Pausa", r"\(\s*Se\b", r"Se o CPF", r"Se o usuário", r"Se o paciente", r"Se ele não", 
            r"Caso tenhamos", r"\[DADOS", r"\[HIST", r"Mensagem atual", r"- Pergunte", r"Agora que temos", r"Se a data"
        ]
        for corte in cortes:
            match = re.search(corte, resposta_candidata, re.IGNORECASE)
            if match:
                resposta_candidata = resposta_candidata[:match.start()].strip()
                
        # Filtro anti-tag prematura: remove a tag se houver pergunta na mesma resposta ou uso de placeholders { }
        if re.search(r"GERAR\\?_AGENDA_GOOGLE", resposta_candidata):
            if "?" in resposta_candidata or "{" in resposta_candidata:
                match = re.search(r"\[?GERAR\\?_AGENDA_GOOGLE\]?", resposta_candidata)
                if match:
                    resposta_candidata = resposta_candidata[:match.start()].strip()
                
        melhor_resposta = resposta_candidata # Salva caso seja a última tentativa
        
        # ==========================================
        # PASSO 2: AVALIAÇÃO DA PRÓPRIA RESPOSTA
        # ==========================================
        prompt_avaliacao = f"""<s>[INST] Atue como um Avaliador de Qualidade Rigoroso.
Sua tarefa é dar uma nota de 0 a 10 para a RESPOSTA DA IA.

REGRAS DE AVALIAÇÃO:
- NOTA 10: A resposta solicitou corretamente um dado que falta, OU pediu a confirmação final resumindo os dados, OU gerou a tag [GERAR_AGENDA_GOOGLE] corretamente após o "sim".
- NOTA 0: A resposta gerou tabelas Markdown de qualquer tipo.
- NOTA 0: A resposta perguntou um dado que JÁ FOI fornecido no Histórico ou Mensagem.
- NOTA 0: A resposta fez perguntas inúteis de confirmação como "O CPF é válido?" ou "Podemos pedir a data?".
- NOTA 0: A resposta tentou gerar blocos de contexto falsos como [DADOS DO BANCO].
- NOTA 0: A resposta contém roteiros, opções ou instruções condicionais (ex: "Se o paciente...", "Se ele não...").
- NOTA 0: A resposta gerou a tag [GERAR_AGENDA_GOOGLE] no mesmo texto de uma pergunta.
- NOTA 0: A resposta gerou a tag [GERAR_AGENDA_GOOGLE] antes de pedir a confirmação final ou antes de o usuário responder "sim".
- NOTA 0: A resposta usou variáveis vazias como {{data}} ou {{horario}}.

[HISTÓRICO DA CONVERSA]
{req.historico}

[MENSAGEM DO USUÁRIO]
{req.pergunta}

[RESPOSTA DA IA]
{resposta_candidata}

Responda APENAS no formato: NOTA: [numero] [/INST]"""
        
        eval_inputs = tokenizer(prompt_avaliacao, return_tensors="pt").to("cuda")
        with torch.no_grad():
            # Removido temperature=0.1 para evitar o conflito com do_sample=False e max_new_tokens aumentado para segurança.
            eval_outputs = model.generate(**eval_inputs, max_new_tokens=10, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        avaliacao_txt = tokenizer.decode(eval_outputs[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        
        # Procura por "NOTA: X" especificamente para evitar pegar números aleatórios da justificativa
        match = re.search(r'(?i)nota\s*[:\-]?\s*(\d+)', avaliacao_txt)
        if match:
            nota = int(match.group(1))
        else:
            numeros = re.findall(r'\d+', avaliacao_txt)
            nota = int(numeros[0]) if numeros else 0
        
        print(f"Candidata: {resposta_candidata}\nAvaliação bruta: {avaliacao_txt}\nNota Avaliada: {nota}")
        
        if nota >= 5:
            break # A nota foi boa o suficiente, sai do loop

    return {"resposta": melhor_resposta}

# Para rodar: uvicorn api_ia:app --host 0.0.0.0 --port 8000