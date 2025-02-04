#!/usr/bin/env python

# Dependencies:
# opencv, do (pip install opencv-python)
# skimage, do (apt install python-skimage)
# termcolor, do (pip install termcolor)

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.optim import AdamW
from Network.Network import HomographyModel
from Network.Network import LossFn
import cv2
import sys
import os
import numpy as np
import random
import skimage
import PIL
import os
import glob
import random
from skimage import data, exposure, img_as_float
import matplotlib.pyplot as plt
import numpy as np
import time
from Misc.MiscUtils import *
from Misc.DataUtils import *
from torchvision.transforms import ToTensor
import argparse
import shutil
import string
from termcolor import colored, cprint
import math as m
from tqdm import tqdm
import wandb

wandb.init(project="Supervised", name="Sup_train2")


def GenerateBatch(DirNamesTrain, MiniBatchSize):
    """
    Inputs:
    DirNamesTrain - List of paths to .npz files
    MiniBatchSize - Batch size
    Outputs:
    PA_Batch - Batch of PA patches (shape: [B, 3, 128, 128])
    PB_Batch - Batch of PB patches (shape: [B, 3, 128, 128])
    H4Pt_Batch - Batch of ground-truth homography parameters (shape: [B, 8])
    """
    # Randomly select .npz files
    SelectedNPZFiles = np.random.choice(DirNamesTrain, MiniBatchSize, replace=False)
    
    PA_Batch = []
    PB_Batch = []
    H4Pt_Batch = []

    for NPZFile in SelectedNPZFiles:
        # Load data from .npz file
        Data = np.load(NPZFile)
        PA = Data["PA"]  # Shape: [128, 128, 3]
        PB = Data["PB"]  # Shape: [128, 128, 3]
        H4Pt = Data["H4Pt"]  # Shape: [8]

        # Append to batch
        PA_Batch.append(PA)
        PB_Batch.append(PB)
        H4Pt_Batch.append(H4Pt)

    # Convert to numpy arrays and transpose to CHW format
    PA_Batch = np.stack(PA_Batch)  # Shape: [B, 128, 128, 3]
    PB_Batch = np.stack(PB_Batch)  # Shape: [B, 128, 128, 3]
    H4Pt_Batch = np.stack(H4Pt_Batch)  # Shape: [B, 8]

    # Transpose from HWC to CHW format
    PA_Batch = np.transpose(PA_Batch, (0, 3, 1, 2))  # Shape: [B, 3, 128, 128]
    PB_Batch = np.transpose(PB_Batch, (0, 3, 1, 2))  # Shape: [B, 3, 128, 128]

    # Convert to PyTorch tensors
    PA_Batch = torch.from_numpy(PA_Batch).float()
    PB_Batch = torch.from_numpy(PB_Batch).float()
    H4Pt_Batch = torch.from_numpy(H4Pt_Batch).float()

    return PA_Batch, PB_Batch, H4Pt_Batch


def PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile):
    """
    Prints all stats with all arguments
    """
    print("Number of Epochs Training will run for " + str(NumEpochs))
    print("Factor of reduction in training data is " + str(DivTrain))
    print("Mini Batch Size " + str(MiniBatchSize))
    print("Number of Training Images " + str(NumTrainSamples))
    if LatestFile is not None:
        print("Loading latest checkpoint with the name " + LatestFile)

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


def TrainOperation(
    DirNamesTrain,
    DirNamesVal,
    TrainCoordinates,
    NumTrainSamples,
    ImageSize,
    NumEpochs,
    MiniBatchSize,
    SaveCheckPoint,
    CheckPointPath,
    DivTrain,
    LatestFile,
    BasePath,
    LogsPath,
    ModelType,
):
    """
    Inputs:
    ImgPH is the Input Image placeholder
    DirNamesTrain - Variable with Subfolder paths to train files
    TrainCoordinates - Coordinates corresponding to Train/Test
    NumTrainSamples - length(Train)
    ImageSize - Size of the image
    NumEpochs - Number of passes through the Train data
    MiniBatchSize is the size of the MiniBatch
    SaveCheckPoint - Save checkpoint every SaveCheckPoint iteration in every epoch, checkpoint saved automatically after every epoch
    CheckPointPath - Path to save checkpoints/model
    DivTrain - Divide the data by this number for Epoch calculation, use if you have a lot of dataor for debugging code
    LatestFile - Latest checkpointfile to continue training
    BasePath - Path to COCO folder without "/" at the end
    LogsPath - Path to save Tensorboard Logs
        ModelType - Supervised or Unsupervised Model
    Outputs:
    Saves Trained network in CheckPointPath and Logs to LogsPath
    """
    # Predict output with forward pass
    model = HomographyModel()

    ###############################################
    # Fill your optimizer of choice here!
    ###############################################
    Optimizer =  torch.optim.Adam(model.parameters(), lr=0.0001)

    # Tensorboard
    # Create a summary to monitor loss tensor
    Writer = SummaryWriter(LogsPath)

    if LatestFile is not None:
        CheckPoint = torch.load(CheckPointPath + LatestFile + ".ckpt")
        # Extract only numbers from the name
        StartEpoch = int("".join(c for c in LatestFile.split("a")[0] if c.isdigit()))
        model.load_state_dict(CheckPoint["model_state_dict"])
        print("Loaded latest checkpoint with the name " + LatestFile + "....")
    else:
        StartEpoch = 0
        print("New model initialized....")

     # Lists to store loss and accuracy values
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for Epochs in tqdm(range(StartEpoch, NumEpochs)):
        NumIterationsPerEpoch = int(NumTrainSamples / MiniBatchSize / DivTrain)
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        for PerEpochCounter in tqdm(range(NumIterationsPerEpoch)):
            # Load batch from .npz files
            PA_Batch, PB_Batch, H4Pt_Batch = GenerateBatch(DirNamesTrain, MiniBatchSize)

            # Forward pass
            PredictedH4Pt = model(PA_Batch, PB_Batch)
            LossThisBatch = LossFn(PredictedH4Pt, H4Pt_Batch)

            # Backward pass and optimization
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Accumulate loss and accuracy for the epoch
            epoch_loss += LossThisBatch.item()
            epoch_accuracy += ComputeAccuracy(PredictedH4Pt, H4Pt_Batch)

        # Compute average loss and accuracy for the epoch
        avg_epoch_loss = epoch_loss / NumIterationsPerEpoch
        avg_epoch_accuracy = epoch_accuracy / NumIterationsPerEpoch

        # Store training loss and accuracy
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(avg_epoch_accuracy)

        wandb.log({"epoch": Epochs, "loss": avg_epoch_loss, "accuracy": avg_epoch_accuracy})

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        NumValSamples = len(DirNamesVal)
        NumValIterations = int(NumValSamples / MiniBatchSize)

        with torch.no_grad():
            for ValCounter in range(NumValIterations):
                # Load validation batch
                PA_Val, PB_Val, H4Pt_Val = GenerateBatch(DirNamesVal, MiniBatchSize)

                # Forward pass
                PredictedH4Pt_Val = model(PA_Val, PB_Val)
                ValLoss = LossFn(PredictedH4Pt_Val, H4Pt_Val)
                ValAcc = ComputeAccuracy(PredictedH4Pt_Val, H4Pt_Val)

                # Accumulate validation loss and accuracy
                val_loss += ValLoss.item()
                val_accuracy += ValAcc

        # Compute average validation loss and accuracy for the epoch
        avg_val_loss = val_loss / NumValIterations
        avg_val_accuracy = val_accuracy / NumValIterations

        # Store validation loss and accuracy
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)

        # Log validation metrics to wandb
        wandb.log({"epoch": Epochs, "val_loss": avg_val_loss, "val_accuracy": avg_val_accuracy})

        if Epochs % 5 == 0:
            SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
            torch.save(
                {
                    "epoch": Epochs,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": Optimizer.state_dict(),
                    "loss": avg_epoch_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": avg_val_accuracy,
                },
                SaveName,
            )
            print("\n" + SaveName + " Model Saved...")

    wandb.finish()


    # Save the loss and accuracy values for later plotting
    np.savez_compressed(
        os.path.join(LogsPath, "training_metrics.npz"),
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        val_losses=val_losses,
        val_accuracies=val_accuracies,
    )
    print("Training metrics saved to training_metrics.npz")


def main():
    """
    Inputs:
    # None
    # Outputs:
    # Runs the Training and testing code based on the Flag
    #"""
    # Parse Command Line arguments
    Parser = argparse.ArgumentParser()
    Parser.add_argument(
        "--BasePath",
        default="data_25k",
        help="Base path of images, Default:/home/lening/workspace/rbe549/YourDirectoryID_p1/Phase2/Data",
    )
    Parser.add_argument(
        "--CheckPointPath",
        default="Checkpoints_sup2/",
        help="Path to save Checkpoints, Default: ../Checkpoints/",
    )

    Parser.add_argument(
        "--ModelType",
        default="Unsup",
        help="Model type, Supervised or Unsupervised? Choose from Sup and Unsup, Default:Unsup",
    )
    Parser.add_argument(
        "--NumEpochs",
        type=int,
        default=21,
        help="Number of Epochs to Train for, Default:50",
    )
    Parser.add_argument(
        "--DivTrain",
        type=int,
        default=1,
        help="Factor to reduce Train data by per epoch, Default:1",
    )
    Parser.add_argument(
        "--MiniBatchSize",
        type=int,
        default=50,
        help="Size of the MiniBatch to use, Default:1",
    )
    Parser.add_argument(
        "--LoadCheckPoint",
        type=int,
        default=0,
        help="Load Model from latest Checkpoint from CheckPointsPath?, Default:0",
    )
    Parser.add_argument(
        "--LogsPath",
        default="Logs_Sup2/",
        help="Path to save Logs for Tensorboard, Default=Logs/",
    )

    Args = Parser.parse_args()
    NumEpochs = Args.NumEpochs
    BasePath = Args.BasePath
    DivTrain = float(Args.DivTrain)
    MiniBatchSize = Args.MiniBatchSize
    LoadCheckPoint = Args.LoadCheckPoint
    CheckPointPath = Args.CheckPointPath
    LogsPath = Args.LogsPath
    ModelType = Args.ModelType

    # Setup all needed parameters including file reading
    (
        DirNamesTrain,
        DirNamesVal,
        SaveCheckPoint,
        ImageSize,
        NumTrainSamples,
        TrainCoordinates,
        NumClasses,
    ) = SetupAll(BasePath, CheckPointPath)

    # Find Latest Checkpoint File
    if LoadCheckPoint == 1:
        LatestFile = FindLatestModel(CheckPointPath)
    else:
        LatestFile = None

    # Pretty print stats
    PrettyPrint(NumEpochs, DivTrain, MiniBatchSize, NumTrainSamples, LatestFile)

    TrainOperation(
        DirNamesTrain,
        DirNamesVal,
        TrainCoordinates,
        NumTrainSamples,
        ImageSize,
        NumEpochs,
        MiniBatchSize,
        SaveCheckPoint,
        CheckPointPath,
        DivTrain,
        LatestFile,
        BasePath,
        LogsPath,
        ModelType,
    )


if __name__ == "__main__":
    main()
