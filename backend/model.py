import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        return x

def load_model():
    model = SimpleMNIST()
    model.load_state_dict(torch.load("mnist_model.pth", map_location="cpu"))
    model.eval()
    return model
