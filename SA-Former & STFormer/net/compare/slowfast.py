import torch
from torch import nn, einsum
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv3d = nn.Conv3d(in_channels=256, out_channels=256,kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=(0, 0, 0))
        self.classifier = nn.Sequential(
            nn.Linear(2048,1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        batch, c, t, h, w = x.size()
        x = F.avg_pool3d(x, (1,8,8))
        x = self.conv3d(x)
        x = x.view(-1,2048)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.rand(2,256,32,8,8)
    model = Model()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    preds = model(x)
    print(preds.shape)









