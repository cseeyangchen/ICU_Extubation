import torch
from torch import nn, einsum
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(512,256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = torch.mean(x,dim=1)
        x = x.view(-1, 512)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.rand(2,7,512)
    model = Model()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    preds = model(x)
    print(preds.shape)
