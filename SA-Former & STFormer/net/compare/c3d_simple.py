import torch
from torch import nn, einsum
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(2, 2, 2))   # 64 31 112 112
        self.pool1 = nn.MaxPool3d(kernel_size=(4, 4, 4), stride=(1,1,1))  # 64 15 28 28

        self.conv11 = nn.Conv3d(64, 64, kernel_size=(3, 3, 3), padding=(2, 2, 2))  # 64 31 112 112
        self.pool11 = nn.MaxPool3d(kernel_size=(4, 4, 4))  # 64 7 7 7

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(2, 2, 2))  # 128 31 112 112
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))   # 128 3 3 3

        self.conv3 = nn.Conv3d(128, 64, kernel_size=(3, 3, 3), padding=(2, 2, 2))  # 64 31 112 112
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))  # 64 1 1 1

        self.conv4 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 64 31 112 112
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2,4,4))  # 64 1 1 1

        self.conv5 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 32 15 28 28
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 4, 4))  # 32 7 7 7

        self.conv6 = nn.Conv3d(32, 32, kernel_size=(3, 3, 3), padding=(1, 1, 1))  # 32 7 7 7
        self.pool6 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))  # 32 3 3 3

        # self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool3 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(3, 3, 3))
        #
        # self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        #
        # self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))

        self.fc5 = nn.Linear(864, 512)
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256,64)
        # self.fc7 = nn.Linear(128,64)
        self.fc8 = nn.Linear(64, 4)
        #
        # self.dropout = nn.Dropout(p=0.5)

        self.relu = nn.ReLU()

    def forward(self, x):
        print("step1:")
        x = self.relu(self.conv1(x))
        print(x.shape)
        x = self.pool1(x)
        print(x.shape)

        print("step1:")
        x = self.relu(self.conv11(x))
        print(x.shape)
        x = self.pool11(x)
        print(x.shape)

        print("step2:")
        x = self.relu(self.conv2(x))
        print(x.shape)
        x = self.pool2(x)
        print(x.shape)

        print("step1:")
        x = self.relu(self.conv3(x))
        print(x.shape)
        x = self.pool3(x)
        print(x.shape)

        print("step1:")
        x = self.relu(self.conv4(x))
        print(x.shape)
        x = self.pool4(x)
        print(x.shape)

        print("step1:")
        x = self.relu(self.conv5(x))
        print(x.shape)
        x = self.pool5(x)
        print(x.shape)

        print("step1:")
        x = self.relu(self.conv6(x))
        print(x.shape)
        x = self.pool6(x)
        print(x.shape)


        # print("step3:")
        # x = self.relu(self.conv3a(x))
        # print(x.shape)
        # x = self.relu(self.conv3b(x))
        # print(x.shape)
        # x = self.pool3(x)
        # print(x.shape)

        # print("step4:")
        # x = self.relu(self.conv4a(x))
        # print(x.shape)
        # x = self.relu(self.conv4b(x))
        # print(x.shape)
        # x = self.pool4(x)
        # print(x.shape)
        #
        # print("step5:")
        # x = self.relu(self.conv5a(x))
        # print(x.shape)
        # x = self.relu(self.conv5b(x))
        # print(x.shape)
        # x = self.pool5(x)
        # print(x.shape)

        # print("step6:")
        # x = x.view(-1, 8192)
        x = x.view(-1, 864)
        x = self.relu(self.fc5(x))
        x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        x = self.relu(self.fc7(x))
        # x = self.dropout(x)

        logits = self.fc8(x)

        return logits

def get_1x_lr_params(model):
    """
    This generator returns all the parameters for conv and two fc layers of the net.
    """
    b = [model.conv1, model.conv2, model.conv3a, model.conv3b, model.fc7]
    for i in range(len(b)):
        for k in b[i].parameters():
            if k.requires_grad:
                yield k

def get_10x_lr_params(model):
    """
    This generator returns all the parameters for the last fc layer of the net.
    """
    b = [model.fc8]
    for j in range(len(b)):
        for k in b[j].parameters():
            if k.requires_grad:
                yield k


if __name__ == "__main__":
    x = torch.rand(1, 3, 31, 112, 112)
    model = Model()
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    preds = model(x)
    print(preds.shape)
