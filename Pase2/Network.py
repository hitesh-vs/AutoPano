"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""

import torch.nn as nn
import sys
import torch
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import kornia  # You can use this to get the transform and warp in this project

# Don't generate pyc codes
sys.dont_write_bytecode = True


# def LossFn(delta, img_a, patch_b, corners):
#     ###############################################
#     # Fill your loss function of choice here!
#     ###############################################

#     ###############################################
#     # You can use kornia to get the transform and warp in this project
#     # Bonus if you implement it yourself
#     ###############################################
#     loss = ...
#     return loss

def LossFn(pred_corners, gt_corners):
    ###############################################
    # Fill your loss function of choice here!
    ###############################################

    ###############################################
    # You can use kornia to get the transform and warp in this project
    # Bonus if you implement it yourself
    ###############################################
    mse_loss = nn.MSELoss(reduction='sum')
    loss = mse_loss(pred_corners, gt_corners)
    return loss

def ComputeAccuracy(PredictedH4Pt, H4Pt_Batch, threshold=5.0):
    """
    Compute accuracy as the percentage of predictions within a threshold.
    Inputs:
    PredictedH4Pt - Predicted homography parameters (shape: [B, 8])
    H4Pt_Batch - Ground truth homography parameters (shape: [B, 8])
    threshold - Error threshold for considering a prediction correct
    Outputs:
    Accuracy - Percentage of predictions within the threshold
    """
    # Compute L2 error between predicted and ground truth
    error = torch.norm(PredictedH4Pt - H4Pt_Batch, dim=1)  # Shape: [B]
    correct = (error < threshold).float()  # 1 if correct, 0 otherwise
    accuracy = correct.mean().item()  # Percentage of correct predictions
    return accuracy


class HomographyModel(pl.LightningModule):
    def __init__(self):
        super(HomographyModel, self).__init__()
        #self.hparams = hparams
        self.model = HomographyNet()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        PA, PB, H4Pt = batch
        PredictedH4Pt = self.model(PA, PB)  # Model processes PA and PB
        loss = F.mse_loss(PredictedH4Pt, H4Pt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        PA, PB, H4Pt = batch
        PredictedH4Pt = self(PA, PB)  # Forward pass
        ValLoss = F.mse_loss(PredictedH4Pt, H4Pt)  # Validation loss
        ValAccuracy = ComputeAccuracy(PredictedH4Pt, H4Pt)  # Validation accuracy
        return {"val_loss": ValLoss, "val_accuracy": ValAccuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


# class Net(nn.Module):
#     def __init__(self, InputSize, OutputSize):
#         """
#         Inputs:
#         InputSize - Size of the Input
#         OutputSize - Size of the Output
#         """
#         super().__init__()
#         #############################
#         # Fill your network initialization of choice here!
#         #############################
#         ...
#         #############################
#         # You will need to change the input size and output
#         # size for your Spatial transformer network layer!
#         #############################
#         # Spatial transformer localization-network
#         self.localization = nn.Sequential(
#             nn.Conv2d(1, 8, kernel_size=7),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#             nn.Conv2d(8, 10, kernel_size=5),
#             nn.MaxPool2d(2, stride=2),
#             nn.ReLU(True),
#         )

#         # Regressor for the 3 * 2 affine matrix
#         self.fc_loc = nn.Sequential(
#             nn.Linear(10 * 3 * 3, 32), nn.ReLU(True), nn.Linear(32, 3 * 2)
#         )

#         # Initialize the weights/bias with identity transformation
#         self.fc_loc[2].weight.data.zero_()
#         self.fc_loc[2].bias.data.copy_(
#             torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
#         )

#     #############################
#     # You will need to change the input size and output
#     # size for your Spatial transformer network layer!
#     #############################
#     def stn(self, x):
#         "Spatial transformer network forward function"
#         xs = self.localization(x)
#         xs = xs.view(-1, 10 * 3 * 3)
#         theta = self.fc_loc(xs)
#         theta = theta.view(-1, 2, 3)

#         grid = F.affine_grid(theta, x.size())
#         x = F.grid_sample(x, grid)

#         return x

#     def forward(self, xa, xb):
#         """
#         Input:
#         xa is a MiniBatch of the image a
#         xb is a MiniBatch of the image b
#         Outputs:
#         out - output of the network
#         """
#         #############################
#         # Fill your network structure of choice here!
#         #############################
#         return out

class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        # VGG-style backbone for 6-channel input (patchA + patchB)
        self.features = nn.Sequential(
            # Layer 1: 6 input channels (RGB for two images)
            nn.Conv2d(6, 64, kernel_size=3, padding=1),  # [n, 6, 128, 128] → [n, 64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # [n, 64, 128, 128] → [n, 64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [n, 64, 64, 64]

            # Layer 2
            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # [n, 64, 64, 64] → [n, 128, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # [n, 128, 64, 64] → [n, 128, 64, 64]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [n, 128, 32, 32]

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # [n, 128, 32, 32] → [n, 256, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # [n, 256, 32, 32] → [n, 256, 32, 32]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [n, 256, 16, 16]

            # Layer 4
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # [n, 256, 16, 16] → [n, 512, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),  # [n, 512, 16, 16] → [n, 512, 16, 16]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # [n, 512, 8, 8]

            nn.Dropout(p=0.5)
        )

        # Fully connected layers for regression
        self.regressor = nn.Sequential(
            nn.Linear(128 * 8 * 8, 1024),  # Adjust based on final feature map size
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 8)  # Output: 8-DOF homography (Δu1, Δv1, ..., Δu4, Δv4)
        )

    def forward(self, xa, xb):
        # Input shapes: [batch_size, 3, 128, 128] for both xa and xb
        # Concatenate along channel dimension
        x = torch.cat([xa, xb], dim=1)  # [batch_size, 6, 128, 128]
        
        # Feature extraction
        x = self.features(x)  # [batch_size, 512, 8, 8]
        
        # Regression
        x = x.reshape(x.shape[0], -1)   # Flatten
        out = self.regressor(x)  # [batch_size, 8]
        
        return out
