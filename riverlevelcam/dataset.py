#%%
import os
import numpy as np
import pandas as pd
import sqlite3
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from skimage import io, transform

class RiverDataset(Dataset):
    def __init__(self, database, root_dir, transform=None, split=None, split_date=1.64e9):
        imgSQL = 'SELECT * from discharge'
        with sqlite3.connect(database) as conn:
            c = conn.cursor()
            c.execute(imgSQL)
            self.data = pd.DataFrame(c.fetchall(), columns=["date", "datestring", "siteno", "gage", "dis", "img"])
            self.data = self.data.sort_values(by="date", ascending=True)
            self.data = self.data.query("gage > 0")
        self.root_dir = root_dir
        self.transform = transform

        self.maxs = self.data.gage.max()
        self.mins = self.data.gage.min()
        self.mus = self.data.gage.mean()
        self.sds = self.data.gage.std()

        if split == "train":
            self.data = self.data.query(f"date < {split_date}")
        elif split == "test":
            self.data = self.data.query(f"date > {split_date}")

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data.iloc[idx]["img"])
        im = io.imread(img_name, format='jpg')
        im = (im[300:,200:,:] / 255.0)
        
        data = (self.data.iloc[idx]["gage"] - self.mus) / self.sds

        sample = {'image': im, 'data': np.array([data])}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def stage(self, x):
        return x * self.sds + self.mus

#%%
class TrivialDataset(Dataset):
    def __init__(self, transform=None, size=10):
        n = int(size/2)
        self.data = np.ones((n,6), dtype=float)
        self.im = np.ones((n,1), dtype=float)

        self.data = np.concatenate([self.data, np.zeros((n,6), dtype=float)])
        self.im = np.concatenate([self.im, np.zeros((n,1), dtype=float)])

        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        im = self.im[idx,:]
        data = self.data[idx, [3,4]].astype(float)#(self.data[idx, [3,4]].astype(float) - self.mins) / (self.maxs-self.mins)
        sample = {'image': im, 'data': data}

        if self.transform:
            sample = self.transform(sample)

        return sample

#%%
class NumpyFlatten:
    def __call__(self, sample):
        return {
            'image': sample['image'].flatten(),
            'data': sample['data']
        }

#%%
class ToTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, data = sample['image'], sample['data']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).type(torch.float32),
                'data': torch.from_numpy(data).type(torch.float32)}

class DummyTensor:
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, data = sample['image'], sample['data']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).type(torch.float),
                'data': torch.from_numpy(data).type(torch.float)}

#%%
class Rescale:
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, data = sample['image'], sample['data']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        #img = img - 0.5
        #img = (img - img.mean()) / img.std()

        return {'image': img, 'data': data}