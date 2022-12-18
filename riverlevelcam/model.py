import pandas as pd
import torch
from torch import nn

class StageModel(nn.Module):
    def __init__(self, inH, inW, inC, outD, hC=2, csize=5, device="cpu"):
        super().__init__()
        self.hC = hC
        self.conv1 = nn.Conv2d(inC, hC, csize)
        hH1 = inH - csize + 1
        hW1 = inW - csize + 1
        self.conv2 = nn.Conv2d(hC, hC, csize)
        hH2 = hH1 - csize + 1
        hW2 = hW1 - csize + 1
        self.conv3 = nn.Conv2d(hC, hC, csize)
        hH3 = hH2 - csize + 1
        hW3 = hW2 - csize + 1
        self.denseSize = hH3 * hW3 * hC

        self.fc1 = nn.Linear(self.denseSize, self.denseSize)
        self.fc2 = nn.Linear(self.denseSize, outD)
        self.relu = nn.ReLU()

        self.device = device
        self.to(device)

    def forward(self, x):
        c1 = self.relu((self.conv1(x)))
        c2 = self.relu((self.conv2(c1)))
        c3 = self.relu((self.conv2(c2)))
        h1 = self.relu(self.fc1(c3.view(-1, self.denseSize)))
        return self.fc2(h1)

    def fit(self, dataloader, optim, epochs):
        # save normalizing constants for predicting new data
        self.mus = dataloader.dataset.mus
        self.sds = dataloader.dataset.sds

        self.train()
        self.losses = []
        for epoch in range(epochs):
            train_loss = 0.0
            n = 0.0
            for data in dataloader:
                imgs = data['image'].to(self.device)
                y = data['data'][:,0][:,None].to(self.device)
                optim.zero_grad()
                y_hat = self.forward(imgs)
                loss = (y_hat - y).pow(2).mean()
                loss.backward()
                train_loss += loss.item()
                n += 1
                optim.step()

            self.losses += [train_loss / n]
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / n ))

    def predict(self, dataloader):
        self.eval() 
        y = []
        y_hat = []
        with torch.no_grad():
            for i, sample in enumerate(dataloader):
                if "data" in sample:
                    norm_data = sample['data'].numpy()[:,0]
                    y += dataloader.dataset.stage(
                        norm_data
                    ).tolist()
                
                image = sample['image'].to(self.device)
                y_hat += self.stage(
                    self.forward(image).cpu().numpy().flatten()
                ).tolist()

        if len(y) == 0:
            y = [np.nan] * len(y_hat)
        return pd.DataFrame({"y": y, "y_hat": y_hat})

    def stage(self, x):
        return x * self.sds + self.mus
