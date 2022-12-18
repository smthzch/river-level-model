import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import torch
import torch.utils.data
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from riverlevelcam.dataset import RiverDataset, ToTensor, Rescale
import riverlevelcam.model

from metaflow import FlowSpec, step, Parameter

class TrainFlow(FlowSpec):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    learning_rate = Parameter("learning_rate", default=1e-4)
    epochs = Parameter("epochs", default=30)
    run_path = Parameter(
        "run_path",
        help="Location to save model and plot outputs.",
        default=f'runs/run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    )
    smoketest = Parameter("smoketest", default=False)

    @step
    def start(self):
        print(f"Running on {self.device}!")

        # check output dir exists
        if not os.path.exists(self.run_path):
            os.makedirs(Path(self.run_path, "plots"))

        # load dataset
        self.transform = transforms.Compose([Rescale(12), ToTensor()])
        self.dataset = RiverDataset("data/data.db", "data/imgs", self.transform, split='train')
        
        if self.smoketest:
            self.dataset.data = self.dataset.data.iloc[0:8]
            self.run_epochs = 1
        else:
            self.run_epochs = self.epochs
        
        self.dataloader = DataLoader(self.dataset, 8, True, num_workers=8)

        self.channels, self.height, self.width = self.dataset[0]["image"].shape

        self.next(self.train)

    @step
    def train(self):
        self.model = riverlevelcam.model.StageModel(
            self.height, 
            self.width, 
            inC=self.channels, 
            outD=1, 
            hC=32, 
            csize=3, 
            device=self.device
        )
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.model.fit(self.dataloader, optimizer, self.run_epochs)

        self.next(self.save)

    @step
    def save(self):
        torch.save(self.model.state_dict(), Path(self.run_path, "model.pt"))

        plt.plot(self.model.losses)
        plt.savefig(Path(self.run_path, "plots/losses.png"))

        self.next(self.validate)

    @step
    def validate(self):
        testset = RiverDataset("data/data.db", "data/imgs", self.transform, split='test')
        testloader = DataLoader(testset, 64, False, num_workers=8)

        preds = self.model.predict(testloader)

        # metrics
        metrics_df = pd.DataFrame({
            "metric": ["mse", "mae", "r2"],
            "value": [
                mean_squared_error(preds.y, preds.y_hat), 
                mean_absolute_error(preds.y, preds.y_hat), 
                r2_score(preds.y, preds.y_hat)
            ]
        })
        metrics_df.to_csv(Path(self.run_path, "metrics.csv"), index=False)

        # true vs pred plot
        plt.scatter(preds.y, preds.y_hat, s=10, alpha=0.5)
        plt.title("Test Set Comparison")
        plt.xlabel("True Level")
        plt.ylabel("Predicted Level")
        plt.savefig(Path(self.run_path, "plots/true_v_predicted.png"), facecolor="white", transparent=False)

        # example images w/ predictions
        minix, maxix = np.argmin(preds.y_hat), np.argmax(preds.y_hat)

        f, ax = plt.subplots(2,1)
        f.tight_layout(h_pad=2)

        for i, ix in enumerate([minix, maxix]):
            dat = testset[ix]
            img = dat["image"].numpy().transpose(1,2,0)
            stage = preds.y[ix]
            predicted = preds.y_hat[ix]
            ax[i].imshow(img)
            ax[i].set_title(f'Stage: {round(stage, 2)}, Pred: {round(predicted, 2)}')

        f.suptitle("Test Set Predictions (Min/Max)")
        plt.subplots_adjust(top=0.85)
        f.savefig(Path(self.run_path, "plots/test_predictions.png"), facecolor="white", transparent=False)

        self.next(self.end)

    @step
    def end(self):
        print("Done.")

if __name__ == "__main__":
    TrainFlow()