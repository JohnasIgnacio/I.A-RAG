import ast
import torch

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)

from peft import (
    LoraConfig,
    get_peft_model
)

from trl import SFTTrainer

# ====================================
# MODELO
# ====================================

model_name = "mistralai/Mistral-7B-Instruct-v0.2"

print("Carregando tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)

tokenizer.pad_token = tokenizer.eos_token

print("Carregando modelo...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="cpu"
)

print("Modelo carregado!")

# ====================================
# DATASET
# ====================================

data = []

with open(
    "data/dataset_secretaria_virtual_5000.json",
    encoding="utf-8"
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue

        item = ast.literal_eval(line)

        pergunta = item["input"]
        resposta = item["output"]

        text = f"""
<s>[INST]
{pergunta}
[/INST]

{resposta}
"""

        data.append({
            "text": text
        })

dataset = Dataset.from_list(data)

# ====================================
# REDUZ DATASET
# ====================================

dataset = dataset.select(
    range(min(600, len(dataset)))
)

print("Dataset carregado!")
print(dataset[0])

# ====================================
# LORA
# ====================================

lora_config = LoraConfig(
    r=4,
    lora_alpha=8,

    target_modules=[
        "q_proj",
        "v_proj"
    ],

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)

model = get_peft_model(
    model,
    lora_config
)

print("LoRA aplicado!")

# ====================================
# TREINO
# ====================================

training_args = TrainingArguments(
    output_dir="./results",

    per_device_train_batch_size=1,

    gradient_accumulation_steps=8,

    learning_rate=2e-4,

    logging_steps=5,

    num_train_epochs=1,

    fp16=False,

    gradient_checkpointing=True,

    save_strategy="epoch"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args
)

# ====================================
# TREINAR
# ====================================

print("Iniciando treino...")

trainer.train()

# ====================================
# SALVAR
# ====================================

trainer.model.save_pretrained(
    "lora_model"
)

tokenizer.save_pretrained(
    "lora_model"
)

print("LoRA salvo com sucesso!")