from torch import nn

class StageModel(nn.Module):
    def __init__(self, inH, inW, inC, outD, hC=2, csize=5):
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

    def forward(self, x):
        c1 = self.relu((self.conv1(x)))
        c2 = self.relu((self.conv2(c1)))
        c3 = self.relu((self.conv2(c2)))
        h1 = self.relu(self.fc1(c3.view(-1, self.denseSize)))
        return self.fc2(h1)
