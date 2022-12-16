#%%
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from itertools import combinations, combinations_with_replacement, permutations, product

from riverlevelcam.dataset import RiverDataset, ToTensor, Rescale
import riverlevelcam.model

#%%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

#%%
TEST = False
DUMMY = False
#%%

transform = transforms.Compose([Rescale(12), ToTensor()])
dataset = RiverDataset("data/data.db", "data/imgs", transform, split='train')
dataloader = DataLoader(dataset, 8, True, num_workers=8)

i, sample = next(enumerate(dataloader))
height = sample['image'].shape[2]
width = sample['image'].shape[3]

# %%
if not TEST:
    f, ax = plt.subplots(4,2)
    for i in range(4):
        for j in range(2):
            ax[i,j].imshow(sample['image'][i*2 + j].detach().numpy().transpose(1,2,0)[:,:,0], cmap="gray")


#%%

def train(epoch, mod, optim, dat, plot=False):
    train_loss = 0.0
    n = 0.0
    for batch_idx, data in enumerate(dat):
        imgs = data['image'].to(device)
        y = data['data'][:,0][:,None].to(device)
        optim.zero_grad()
        y_hat = mod(imgs)
        loss = (y_hat - y).pow(2).mean()
        loss.backward()
        train_loss += loss.item()
        n += 1
        optim.step()
        
    print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss/n ))

    return train_loss/n

#%%
model = riverlevelcam.model.StageModel(height, width, inC=3, outD=1, hC=32, csize=3)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
losses = []

#%%
nstep = 1
model.train()
epochs = nstep
for i in range(epochs):
    losses += [train(i, model, optimizer, dataloader, plot=False)]

#%%
model_path = 'params/model.pt'
torch.save(model.state_dict(), model_path)

#%%
model.load_state_dict(torch.load('params/model.pt'))
model.eval()

#%%
plt.plot(losses)

#%%
if not TEST:
    #i, sample = next(enumerate(dataloader))
    model.eval()

    #compare model to original
    f, ax = plt.subplots(4,1)
    for i, sample in enumerate(dataloader):
        if i==4: break
        img = sample['image'][0].detach().numpy().transpose(1,2,0)[:,:,0]
        stage = dataset.stage(sample["data"][0].detach().numpy()[0])
        predicted = dataset.stage(model(sample["image"].to(device)).detach().cpu().numpy()[0,0])
        ax[i].imshow(img, cmap="gray", vmin=0,vmax=1)
        ax[i].set_title(f'Stage: {round(stage, 2)}, Predicted: {round(predicted,2)}')

#%%
testset = RiverDataset("data/data.db", "data/imgs", transform, split='test')
testloader = DataLoader(dataset, 64, False, num_workers=8)

model.eval()
y = []
y_hat = []
for i, sample in enumerate(testloader):
    y += dataset.stage(sample['data'].detach().numpy()[:,0]).tolist()
    y_hat += dataset.stage(model(sample['image'].to(device)).detach().cpu().numpy().flatten()).tolist()

#%%
plt.scatter(y, y_hat, s=10, alpha=0.5)
plt.title("Test Set Comparison")
plt.xlabel("True Level")
plt.ylabel("Predicted Level")
plt.savefig("true_v_predicted.png", facecolor="white")

# %%
resid = np.array(y_hat) - np.array(y)
# %%
plt.scatter(x=y, y=resid)
# %%
np.abs(resid).mean()
# %%
minix, maxix = np.argmin(y_hat), np.argmax(y_hat)

f, ax = plt.subplots(2,1)
f.tight_layout(h_pad=2)

for i, ix in enumerate([minix, maxix]):
    dat = testset[ix]
    img = dat["image"].detach().numpy().transpose(1,2,0)
    stage = y[ix]
    predicted = y_hat[ix]
    ax[i].imshow(img)
    ax[i].set_title(f'Stage: {round(stage, 2)}, Pred: {round(predicted, 2)}')

f.suptitle("Test Set Predictions (Min/Max)")
plt.subplots_adjust(top=0.85)
f.savefig("test_predictions.png", facecolor="white")

# %%
