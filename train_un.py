#!/usr/bin/env python

"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano: Phase 2 Starter Code


Author(s):
Lening Li (lli4@wpi.edu)
Teaching Assistant in Robotics Engineering,
Worcester Polytechnic Institute
"""


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
from Network.Network_unsupervised import SpatialTransformationLayer
from Network.Network_unsupervised import LossFn,tensor_dlt,photometric_loss,target_patch_generation
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

wandb.init(project="Supervised", name="Unsup")

def GenerateBatch(DirNamesTrain, MiniBatchSize):

    # Randomly select .npz files
    SelectedNPZFiles = np.random.choice(DirNamesTrain, MiniBatchSize, replace=False)
    
    PA_Batch = []
    PB_Batch = []
    H4Pt_Batch = []
    CAPt_Batch = []

    for NPZFile in SelectedNPZFiles:
        # Load data from .npz file
        Data = np.load(NPZFile)
        PA = Data["PA"]  # Shape: [N, 128, 128, 3]
        PB = Data["PB"]  # Shape: [N, 128, 128, 3]
        H4Pt = Data["H4Pt"]  # Shape: [N, 8]

        CAPt = Data["CA"]    ## new part

        # Randomly select a sample from this .npz file
        #RandIdx = np.random.randint(0, PA.shape[0])
        PA_Batch.append(PA)
        PB_Batch.append(PB)
        H4Pt_Batch.append(H4Pt)
        CAPt_Batch.append(CAPt)

    # Convert to tensors
    PA_Batch = torch.from_numpy(np.stack(PA_Batch)).float()
    PB_Batch = torch.from_numpy(np.stack(PB_Batch)).float()
    H4Pt_Batch = torch.from_numpy(np.stack(H4Pt_Batch)).float()
    CAPt_Batch = torch.from_numpy(np.stack(CAPt_Batch)).float().squeeze()

    PA_Batch = PA_Batch.permute(0, 3, 1, 2)
    PB_Batch = PB_Batch.permute(0, 3, 1, 2)

    return PA_Batch, PB_Batch, H4Pt_Batch, CAPt_Batch


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

    # Predict output with forward pass
    model = HomographyModel()

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
            PA_Batch, PB_Batch, H4Pt_Batch, CAPt_Batch = GenerateBatch(DirNamesTrain, MiniBatchSize)

            height, width = PA_Batch.shape[2], PA_Batch.shape[3]

            # Forward pass
            PredictedH4Pt = model(PA_Batch, PB_Batch)

            CBPt_Batch = target_patch_generation(CAPt_Batch,PredictedH4Pt)

            homography_estimate = tensor_dlt(CAPt_Batch,CBPt_Batch)
            transform_layer = SpatialTransformationLayer()

            warped_image = transform_layer(PA_Batch, homography_estimate, (height, width))

            LossThisBatch = photometric_loss(warped_image,PB_Batch)

            # LossThisBatch = LossFn(PredictedH4Pt, H4Pt_Batch) 

            # Backward pass and optimization
            Optimizer.zero_grad()
            LossThisBatch.backward()
            Optimizer.step()

            # Accumulate loss and accuracy for the epoch
            epoch_loss += LossThisBatch.item()
            epoch_accuracy += ComputeAccuracy(PredictedH4Pt, H4Pt_Batch)   # Check this 

            # # Save checkpoint
            # if PerEpochCounter % SaveCheckPoint == 0:
            #     SaveName = CheckPointPath + str(Epochs) + "a" + str(PerEpochCounter) + "model.ckpt"
            #     torch.save(
            #         {
            #             "epoch": Epochs,
            #             "model_state_dict": model.state_dict(),
            #             "optimizer_state_dict": Optimizer.state_dict(),
            #             "loss": LossThisBatch,
            #         },
            #         SaveName,
            #     )
            #     print("\n" + SaveName + " Model Saved...")

        # Compute average loss and accuracy for the epoch
        avg_epoch_loss = epoch_loss / NumIterationsPerEpoch
        avg_epoch_accuracy = epoch_accuracy / NumIterationsPerEpoch

        # Store training loss and accuracy
        train_losses.append(avg_epoch_loss)
        train_accuracies.append(avg_epoch_accuracy)

        wandb.log({"epoch_unsup": Epochs, "loss_unsup": avg_epoch_loss, "acc_unsup": avg_epoch_accuracy})

       # Validation loop
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        NumValSamples = len(DirNamesVal)
        NumValIterations = int(NumValSamples / MiniBatchSize)

        with torch.no_grad():
            for ValCounter in range(NumValIterations):
                # Load validation batch
                PA_Val, PB_Val, H4Pt_Val, CAPt_Val = GenerateBatch(DirNamesVal, MiniBatchSize)

                # Forward pass and compute metrics using validation_step
                PredictedH4Pt_Val = model(PA_Val, PB_Val)
                CBPt_Batch_Val = target_patch_generation(CAPt_Val, PredictedH4Pt_Val)
        
                height, width = PA_Val.shape[2], PA_Val.shape[3]
                homography_estimate = tensor_dlt(CAPt_Val, CBPt_Batch_Val)
                transform_layer = SpatialTransformationLayer()
                warped_image = transform_layer(PA_Val, homography_estimate, (height, width))
        
                # Compute loss and accuracy
                ValLoss = photometric_loss(warped_image, PB_Val)
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

        # Log metrics
        wandb.log({"val_loss_unsup": avg_val_loss, "val_acc_unsup": avg_val_accuracy})

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

        # # Save model every epoch
        # SaveName = CheckPointPath + str(Epochs) + "model.ckpt"
        # torch.save(
        #     {
        #         "epoch": Epochs,
        #         "model_state_dict": model.state_dict(),
        #         "optimizer_state_dict": Optimizer.state_dict(),
        #         "loss": avg_epoch_loss,
        #     },
        #     SaveName,
        # )
        # print("\n" + SaveName + " Model Saved...")

    
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
        default="Checkpoints_unsup/",
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
        default=30,
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
        default="Logs_unsup/",
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
