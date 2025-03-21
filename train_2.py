from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from my_model import GatedDeltaNet
from tqdm import tqdm  # Для красивого прогресс-бара
import math

# Инициализируем токенизатора
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Добавляем специальный токен для паддинга

# Параметры
batch_size = 8
sequence_length = 128
hidden_size = 1024
vocab_size = tokenizer.vocab_size  # Размер словаря определяется токенизатором
num_epochs = 1
learning_rate = 1e-4

# Загружаем датасет
dataset = load_dataset("tiny_shakespeare", trust_remote_code=True)

# Функция для токенизации текста
def tokenize_function(texts):
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=sequence_length,  # Ограничение длины последовательности
        return_tensors="pt"
    )

# Токенизируем обучающие и валидационные данные
train_encodings = tokenize_function(dataset['train']['text'])
validation_encodings = tokenize_function(dataset['validation']['text'])

# Преобразуем в тензоры
X_train = train_encodings["input_ids"]
y_train = torch.cat([X_train[:, 1:], torch.zeros((X_train.size(0), 1), dtype=torch.long)], dim=1)

X_val = validation_encodings["input_ids"]
y_val = torch.cat([X_val[:, 1:], torch.zeros((X_val.size(0), 1), dtype=torch.long)], dim=1)

# Создаем Dataset и DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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

# Определение loss и оптимизатора
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Цикл обучения
device = torch.device("cpu")
model.to(device)

# Цикл обучения и оценки
for epoch in range(num_epochs):
    # *** Обучение ***
    model.train()
    train_running_loss = 0.0
    train_total_batches = len(train_loader)
    train_progress_bar = tqdm(enumerate(train_loader), total=train_total_batches, desc=f"Train Epoch {epoch+1}/{num_epochs}")

    for batch_idx, (inputs, targets) in train_progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs, _, _ = model(inputs)  # Выход: (batch_size, sequence_length, vocab_size)
        assert not torch.isnan(outputs).any(), "Model outputs contain NaN"
        assert not torch.isinf(outputs).any(), "Model outputs contain Inf"
        outputs = outputs.view(-1, vocab_size)  # (batch_size * sequence_length, vocab_size)
        targets = targets.view(-1)             # (batch_size * sequence_length)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Обновление running_loss
        train_running_loss += loss.item()

        # Вывод прогресса каждые N батчей
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == train_total_batches:
            avg_loss = train_running_loss / (batch_idx + 1)
            train_progress_bar.set_postfix({"Train Loss": f"{avg_loss:.4f}"})

    train_epoch_loss = train_running_loss / train_total_batches

    # *** Оценка ***
    model.eval()
    val_running_loss = 0.0
    val_total_batches = len(val_loader)
    val_progress_bar = tqdm(enumerate(val_loader), total=val_total_batches, desc=f"Val Epoch {epoch+1}/{num_epochs}")

    with torch.no_grad():  # Отключаем вычисление градиентов
        for batch_idx, (inputs, targets) in val_progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs, _, _ = model(inputs)  # Выход: (batch_size, sequence_length, vocab_size)
            outputs = outputs.view(-1, vocab_size)  # (batch_size * sequence_length, vocab_size)
            targets = targets.view(-1)             # (batch_size * sequence_length)

            # Compute loss
            loss = criterion(outputs, targets)

            # Обновление running_loss
            val_running_loss += loss.item()

            # Вывод прогресса каждые N батчей
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == val_total_batches:
                avg_loss = val_running_loss / (batch_idx + 1)
                val_progress_bar.set_postfix({"Val Loss": f"{avg_loss:.4f}"})

    val_epoch_loss = val_running_loss / val_total_batches

    # Вывод итогов эпохи
    print(f"Epoch [{epoch+1}/{num_epochs}] completed.")
    print(f"  Train Loss: {train_epoch_loss:.4f}, Perplexity: {math.exp(train_epoch_loss):.4f}")
    print(f"  Val Loss: {val_epoch_loss:.4f}, Perplexity: {math.exp(val_epoch_loss):.4f}")

torch.save(model.state_dict(), "gated_delta_net_2.pth")