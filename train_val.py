import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from my_model import GatedDeltaNet
from tqdm import tqdm  # Для красивого прогресс-бара
import math

# Параметры
batch_size = 8
sequence_length = 128
hidden_size = 1024
vocab_size = 50257
num_samples = 1000
num_epochs = 1
learning_rate = 1e-4

# Генерация данных для CLM
X = torch.randint(0, vocab_size, (num_samples, sequence_length))  # Входные данные
y = torch.cat([X[:, 1:], torch.zeros((num_samples, 1), dtype=torch.long)], dim=1)  # Целевые данные

# Создание DataLoader
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
criterion = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Цикл обучения
device = torch.device("cpu")
model.to(device)

# Разделение данных на обучающую и валидационную выборки
train_size = int(0.8 * len(dataset))  # 80% данных для обучения
val_size = len(dataset) - train_size  # 20% данных для валидации
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Создание DataLoader для обучающей и валидационной выборок
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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