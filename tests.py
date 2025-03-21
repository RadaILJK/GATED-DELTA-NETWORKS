import torch
import torch.nn as nn
from transformers import AutoTokenizer
from my_model import GatedDeltaNet
from datasets import load_dataset

# Загрузка датасета
dataset = load_dataset("tiny_shakespeare")

# Инициализация токенизатора
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Добавляем специальный токен для паддинга

# Параметры
sequence_length = 128
batch_size = 4
hidden_size = 1024
vocab_size = tokenizer.vocab_size

# Загрузка тестовых данных
test_data = dataset['test']['text'][:batch_size]  # Берем несколько примеров для тестирования


# Токенизация тестовых данных
def tokenize_function(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,
        return_tensors="pt"
    )


test_encodings = tokenize_function(test_data)
test_inputs = test_encodings["input_ids"]  # (batch_size, sequence_length)

# Инициализация модели
model = GatedDeltaNet(
    mode='chunk',
    hidden_size=hidden_size,
    expand_k=0.75,
    expand_v=1.5,
    num_heads=8,
    qk_norm='l2',
    conv_size=4,
    gate_fn='swish',
    use_mamba_gate=True,
    use_residual=True,
)
model.lm_head = nn.Linear(hidden_size, vocab_size)  # Добавляем слой для выхода LM

# Загрузка сохраненных весов
model.load_state_dict(torch.load("gated_delta_net_2.pth", map_location=torch.device('cpu')))
model.eval()  # Переводим модель в режим оценки


# Функция для генерации текста
def generate_text(model, input_ids, max_new_tokens=50):
    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Обрезаем входные данные до последних `sequence_length` токенов
            input_ids = input_ids[:, -sequence_length:]

            # Получаем предсказания модели
            outputs, _, _ = model(input_ids)
            next_token_logits = outputs[:, -1, :]  # (batch_size, vocab_size)
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(
                -1)  # Выбираем токен с наибольшей вероятностью

            # Добавляем новый токен к входным данным
            input_ids = torch.cat([input_ids, next_token_id], dim=-1)

        # Декодируем токены обратно в текст
        generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return generated_text


# Тестирование модели
print("Тестовые примеры:")
for i, text in enumerate(test_data):
    print(f"Пример {i + 1}:")
    print(f"Исходный текст: {text[:100]}...")  # Показываем первые 100 символов

    # Генерация текста
    input_ids = test_inputs[i].unsqueeze(0)  # (1, sequence_length)
    generated_text = generate_text(model, input_ids)

    print(f"Сгенерированный текст: {generated_text}")
    print("-" * 80)