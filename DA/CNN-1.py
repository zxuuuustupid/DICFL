import torch
import torch.nn as nn


class CNN1DModel(nn.Module):
    def __init__(self):
        super(CNN1DModel, self).__init__()
        self.feature_encoding = nn.Sequential(

            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=64, padding=32, stride=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),


            nn.Conv1d(in_channels=32, out_channels=48, kernel_size=16, padding=8, stride=1),
            nn.BatchNorm1d(48),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),


            nn.Conv1d(in_channels=48, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),


            nn.Conv1d(in_channels=64, out_channels=64, kernel_size=5, padding=2, stride=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        self.classification = nn.Sequential(
            nn.Linear(8192, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.feature_encoding(x)
        x = x.view(x.size(0), -1)
        x = self.classification(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN1DModel().to(device)


# print(model)

# model = CNN1DModel().to(device)
#
input_data = torch.randn(1, 1, 2048).to(device)


output = model(input_data)

print("Output shape:", output.shape)
