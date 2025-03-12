import torch
from my_model import GatedDeltaNet

model = GatedDeltaNet(hidden_size=1024, num_heads=8)
hidden_states = torch.randn(2, 128, 1024)  # (batch_size, seq_len, hidden_size)
output, _, _ = model(hidden_states)
output = output[:, :128, :]
print(output.shape)  # Ожидаемый размер: [2, 128, 1024]
print(output[0])