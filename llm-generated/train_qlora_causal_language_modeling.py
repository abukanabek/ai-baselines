"""
train_qlora_causal.py
Простой QLoRA-пайплайн для CAUSAL_LM (инструкционный стиль).
Подходит для 7B-class моделей на A100 20GB.
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    logging as hf_logging,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

hf_logging.set_verbosity_info()

# -------------------------
# Параметры — подкорректируй по нуждам
# -------------------------
MODEL_NAME = "qwen/medium"  # <- замени на реальный HF id, например "qwen/qwen-7b" или "mistralai/Mistral-7B-Instruct"
DATA_PATH_OR_ID = "data/train.jsonl"  # либо HF dataset id, либо путь к jsonl/csv
OUTPUT_DIR = "./qlora_out"
MAX_SEQ_LENGTH = 1024
PER_DEVICE_BATCH = 4            # per-device batch (маленький для 4-bit)
GRAD_ACCUM = 8                  # effective batch = PER_DEVICE_BATCH * GRAD_ACCUM
EPOCHS = 3
LEARNING_RATE = 2e-4
SEED = 42
SAVE_STEPS = 500
LOGGING_STEPS = 50

# LoRA параметры
LORA_R = 8
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
# Часто встречающиеся имена модулей внимания. Если у модели другие имена — проверь model.named_modules()
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj", "gate_proj", "down_proj", "up_proj"]

# BitsAndBytes (QLoRA) конфиг
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",            # nf4 обычно лучше для LLM
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 # A100 поддерживает bf16; если нет — попробуй torch.float16
)

# -------------------------
# Утилиты
# -------------------------
def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Токенайзер и модель (4-bit)
# -------------------------
print("Загрузка токенайзера...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
# для causal lm желательно иметь pad_token
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Загрузка модели (4-bit QLoRA)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,   # некоторые community-модели требуют этого
)

# Подготовка модели для k-bit тренировки
model = prepare_model_for_kbit_training(model)

# -------------------------
# Прикручиваем LoRA (PEFT)
# -------------------------
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora_config)

def count_trainable(m):
    t = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in m.parameters())
    return t, tot, t / tot

trainable, total, frac = count_trainable(model)
print(f"Trainable params: {trainable:,} / {total:,} ({frac:.4%})")

# -------------------------
# Датасет
# Ожидается jsonl с полем 'text' или пара 'prompt'+'response'.
# -------------------------
print("Загружаем датасет...")
if os.path.exists(DATA_PATH_OR_ID):
    # допустим JSONL или CSV local
    if DATA_PATH_OR_ID.endswith(".jsonl") or DATA_PATH_OR_ID.endswith(".json"):
        ds = load_dataset("json", data_files=DATA_PATH_OR_ID, split="train")
    elif DATA_PATH_OR_ID.endswith(".csv"):
        ds = load_dataset("csv", data_files=DATA_PATH_OR_ID, split="train")
    else:
        ds = load_dataset(DATA_PATH_OR_ID, split="train")
else:
    # treat as HF dataset id
    ds = load_dataset(DATA_PATH_OR_ID, split="train")

# Нормализация примера в поле 'text'
def make_text(example):
    if "prompt" in example and "response" in example:
        t = example["prompt"].strip() + "\n\n" + example["response"].strip()
    elif "text" in example:
        t = example["text"].strip()
    else:
        # fallback: join всех полей
        t = " ".join(str(v) for v in example.values())
    return {"text": t}

ds = ds.map(make_text, remove_columns=ds.column_names)

# Токенизация
def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False)

ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# Data collator для causal LM
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------
# TrainingArguments + Trainer
# -------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=PER_DEVICE_BATCH,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=False,         # мы используем bfloat16 через bitsandbytes compute dtype
    bf16=True,          # A100 поддерживает bf16
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    remove_unused_columns=False,
    report_to="none",
    seed=SEED,
    save_strategy="steps",
    dataloader_pin_memory=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds,
    data_collator=data_collator,
)

# -------------------------
# Запуск тренировки
# -------------------------
print("Старт тренировки QLoRA...")
trainer.train()
print("Тренировка завершена.")

# Сохраняем adapter (только lora-адаптер)
print("Сохраняем PEFT adapter...")
model.save_pretrained(os.path.join(OUTPUT_DIR, "lora_adapter"))

# -------------------------
# Короткий пример инференса с загруженным адаптером
# -------------------------
print("Проверка инференса (загружаем базовую модель + adapter)...")
# Загружаем базовую модель в 4-bit и прикручиваем адаптер
inference_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
)
inference_model = get_peft_model(inference_model, lora_config)
# загрузка adapter (PEFT сохранит pytorch_model.bin внутри папки)
inference_model.load_state_dict(
    torch.load(os.path.join(OUTPUT_DIR, "lora_adapter", "pytorch_model.bin")),
    strict=False,
)
inference_model.eval()

prompt = "Кратко изложи суть: Почему нейросети важны?"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(next(inference_model.parameters()).device)

with torch.no_grad():
    gen = inference_model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=False,
        temperature=0.0,
        top_p=0.95,
        repetition_penalty=1.05,
        pad_token_id=tokenizer.eos_token_id,
    )
print("=== Результат ===")
print(tokenizer.decode(gen[0], skip_special_tokens=True))

print("Готово. Адаптер лежит в:", os.path.join(OUTPUT_DIR, "lora_adapter"))
