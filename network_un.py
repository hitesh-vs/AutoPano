

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


# import kornia  # You can use this to get the transform and warp in this project

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
import torch

def photometric_loss(warped_image, target_image, mask=None):
    """
    Computes photometric loss (Mean Squared Error) between a warped image and the target image.
    
    Args:
        warped_image (torch.Tensor): Transformed source image of shape (B, C, H, W).
        target_image (torch.Tensor): Target image of shape (B, C, H, W).
        mask (torch.Tensor, optional): Binary mask of shape (B, 1, H, W) indicating valid pixels.
    
    Returns:
        torch.Tensor: Scalar photometric loss.
    """
    diff = warped_image - target_image
    loss = (diff ** 2).mean(dim=1, keepdim=True)  # MSE per pixel (ignoring color channels)

    if mask is not None:
        loss = (loss * mask).sum() / (mask.sum() + 1e-8)  # Avoid division by zero
    else:
        loss = loss.mean()

    return loss

def LossFn(pred_corners, gt_corners):

    mse_loss = nn.MSELoss(reduction='sum')
    loss = mse_loss(pred_corners, gt_corners)
    return loss

def ComputeAccuracy(PredictedH4Pt, H4Pt_Batch, threshold=5.0):

    # Compute L2 error between predicted and ground truth
    error = torch.norm(PredictedH4Pt - H4Pt_Batch, dim=1)  # Shape: [B]
    correct = (error < threshold).float()  # 1 if correct, 0 otherwise
    accuracy = correct.mean().item()  # Percentage of correct predictions
    return accuracy

def tensor_dlt(src_pts, dst_pts):
    batch_size = src_pts.shape[0]

    # Extract individual coordinates
    x, y = src_pts[:, :, 0], src_pts[:, :, 1]  # (B, 4)
    x_prime, y_prime = dst_pts[:, :, 0], dst_pts[:, :, 1]  # (B, 4)

    # Construct the matrix A_hat (batch-wise)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)

    # Construct the A matrix as 8x8 system (removing the last column)
    A = torch.stack([
        x, y, ones, zeros, zeros, zeros, -x_prime * x, -x_prime * y,
        zeros, zeros, zeros, x, y, ones, -y_prime * x, -y_prime * y
    ], dim=-1).reshape(batch_size, 8, 8)

    # Construct vector b (the negative of what would have been the last column)
    b = torch.stack([x_prime, y_prime], dim=-1).reshape(batch_size, 8, 1)

    # Solve for h_hat using pseudo-inverse
    try:
        h_hat = torch.linalg.solve(A, b)  # Try direct solve first
    except:
        # Fall back to pseudo-inverse if direct solve fails
        A_pinv = torch.linalg.pinv(A)
        h_hat = A_pinv @ b

    # Reshape h_hat to get the first 8 elements of homography
    h_hat = h_hat.squeeze(-1)  # Shape: (B, 8)

    # Append 1 for the last element (H33 = 1)
    h = torch.cat([h_hat, torch.ones((batch_size, 1), device=src_pts.device)], dim=-1)

    # Reshape to 3x3 homography matrix
    homography = h.view(batch_size, 3, 3)

    # Normalize the homography matrix
    homography = homography / homography[:, -1:, -1:]

    return homography

def target_patch_generation(CAPt_Batch,PredictedH4Pt):

    
    B = CAPt_Batch.shape[0]  # Batch size

    # Reshape H4Pt from (B, 8) -> (B, 4, 2) to match CA
    H4Pt_reshaped = PredictedH4Pt.view(B, 4, 2)

    # Compute CB = CA + H4Pt
    CBPt_Batch = CAPt_Batch + H4Pt_reshaped
    print("CB shape:", CBPt_Batch.shape)  # Should be (B, 4, 2)

    return CBPt_Batch

class SpatialTransformationLayer(nn.Module):
    def __init__(self):
        super(SpatialTransformationLayer, self).__init__()

    def compute_normalized_inverse(self, H, width, height):
        """
        Compute the normalized inverse homography H˜inv = M^(-1) * H^(-1) * M
        """
        batch_size = H.shape[0]
        device = H.device

        # Create normalization matrix M
        M = torch.zeros(batch_size, 3, 3, device=device)
        M[:, 0, 0] = width / 2
        M[:, 0, 2] = width / 2
        M[:, 1, 1] = height / 2
        M[:, 1, 2] = height / 2
        M[:, 2, 2] = 1

        # Compute M inverse
        M_inv = torch.zeros_like(M)
        M_inv[:, 0, 0] = 2 / width
        M_inv[:, 0, 2] = -1
        M_inv[:, 1, 1] = 2 / height
        M_inv[:, 1, 2] = -1
        M_inv[:, 2, 2] = 1

        # Compute inverse of H
        H_inv = torch.linalg.inv(H)

        # Compute normalized inverse homography
        H_inv_norm = M_inv @ H_inv @ M

        return H_inv_norm

    def generate_sampling_grid(self, H_inv, image_A_size, image_B_size):
        """
        Parameterized Sampling Grid Generator (PSGG)
        Args:
            H_inv: inverse homography matrix (B, 3, 3)
            image_A_size: tuple of (height, width) for source image
            image_B_size: tuple of (height, width) for target image
        Returns:
            sampling_points: coordinates in image A for each pixel in image B
        """
        B = H_inv.shape[0]
        H0, W0 = image_B_size
        device = H_inv.device

        # Create grid for image B
        x = torch.linspace(0, W0-1, W0, device=device)
        y = torch.linspace(0, H0-1, H0, device=device)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')

        # Reshape to homogeneous coordinates
        ones = torch.ones_like(x_grid)
        grid = torch.stack([x_grid, y_grid, ones], dim=-1)  # (H0, W0, 3)
        grid = grid.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H0, W0, 3)

        # Apply inverse homography
        grid = grid.view(B, H0*W0, 3, 1)
        transformed = H_inv.unsqueeze(1) @ grid  # (B, H0*W0, 3, 1)
        transformed = transformed.squeeze(-1)  # (B, H0*W0, 3)

        # Convert to 2D coordinates
        transformed = transformed[..., :2] / (transformed[..., 2:] + 1e-8)
        sampling_points = transformed.view(B, H0, W0, 2)

        # Normalize to [-1, 1] for grid_sample
        sampling_points[..., 0] = 2 * sampling_points[..., 0] / (image_A_size[1] - 1) - 1
        sampling_points[..., 1] = 2 * sampling_points[..., 1] / (image_A_size[0] - 1) - 1

        return sampling_points

    def bilinear_sample(self, image, grid):
        """
        Differentiable Sampling (DS) with bilinear interpolation
        Implements equation (11) from the paper using PyTorch's grid_sample
        """
        return F.grid_sample(
            image, 
            grid, 
            mode='bilinear', 
            padding_mode='zeros',
            align_corners=True
        )

    def forward(self, image_A, H, image_B_size):

        # Compute normalized inverse homography
        H_inv_norm = self.compute_normalized_inverse(H, image_B_size[1], image_B_size[0])

        # Generate sampling grid using PSGG
        sampling_grid = self.generate_sampling_grid(
            H_inv_norm, 
            (image_A.shape[2], image_A.shape[3]),  # image_A size
            image_B_size
        )

        # Apply differentiable sampling
        warped_image = self.bilinear_sample(image_A, sampling_grid)

        return warped_image


class HomographyModel(nn.Module):
    def __init__(self):
        super(HomographyModel, self).__init__()
        #self.hparams = hparams
        self.model = EnhancedHomographyNet()

    def forward(self, a, b):
        return self.model(a, b)

    def training_step(self, batch, batch_idx):
        PA, PB, H4Pt = batch
        PredictedH4Pt = self.model(PA, PB)  # Model processes PA and PB
        loss = F.mse_loss(PredictedH4Pt, H4Pt)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        PA, PB, H4Pt,CAPt = batch
        PredictedH4Pt = self(PA, PB)  # Forward pass
        CBPt_Batch = target_patch_generation(CAPt,PredictedH4Pt)
        
        height, width = PA.shape[2], PA.shape[3]

        homography_estimate = tensor_dlt(CAPt,CBPt_Batch)
        transform_layer = SpatialTransformationLayer()

        warped_image = transform_layer(PA, homography_estimate, (height, width))

        ValLoss = photometric_loss(warped_image,PB)
        # ValLoss = F.mse_loss(PredictedH4Pt, H4Pt)  # Validation loss 
        ValAccuracy = ComputeAccuracy(PredictedH4Pt, H4Pt)  # Validation accuracy
        return {"val_loss": ValLoss, "val_accuracy": ValAccuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": logs}


class SpatialTransformerModule(nn.Module):
    def __init__(self, input_channels):
        super(SpatialTransformerModule, self).__init__()
        
        # Localization network - modified for higher input channels
        self.localization = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )

        # Use a dummy tensor to calculate the size dynamically
        dummy_input = torch.randn(1, input_channels, 128, 128)  # Assuming input size of 128x128
        self.fc_size = self._get_conv_output(dummy_input)  # Calculate the output size

        # Regressor for the transformation matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.fc_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 8)  # 8 parameters for homography
        )

        # Initialize with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.zero_()  # Initialize with zero displacement

    def _get_conv_output(self, shape):
        """ Helper function to calculate the output shape of the convolutional layers """
        dummy_out = self.localization(shape)
        return dummy_out.numel()  # Get the number of elements in the tensor after the conv layers

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.fc_size)
        theta = self.fc_loc(xs)
        return theta


class EnhancedHomographyNet(nn.Module):
    def __init__(self):
        super(EnhancedHomographyNet, self).__init__()
        
        # Feature extraction backbone
        self.features = nn.Sequential(
            # Layer 1: 6 input channels (RGB for two images)
            nn.Conv2d(6, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Layer 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Spatial Transformer Network for feature alignment
        self.stn = SpatialTransformerModule(input_channels=6)

        # Final regressor
        self.regressor = nn.Sequential(
            nn.Linear(512 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),  # Added dropout for regularization
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 8)  # Output: 8-DOF homography (Δu1, Δv1, ..., Δu4, Δv4)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, xa, xb):
        # Input processing
        x = torch.cat([xa, xb], dim=1)  # Assuming input is already in NCHW format
        
        # Get initial transformation from STN
        theta_initial = self.stn(x)
        
        # Extract features
        features = self.features(x)
        features = features.reshape(features.size(0), -1)
        
        # Final regression with residual connection
        theta_residual = self.regressor(features)
        
        # Combine initial and residual predictions
        theta_final = theta_initial + theta_residual
        
        return theta_final

# Example usage
if __name__ == "__main__":
    # Create sample input
    batch_size = 2
    height, width = 128, 128
    channels = 3
    
    xa = torch.randn(batch_size, channels, height, width)
    xb = torch.randn(batch_size, channels, height, width)
    
    # Initialize model
    model = EnhancedHomographyNet()
    
    # Forward pass
    output = model(xa, xb)
    print("Output shape:", output.shape)  # Should be [batch_size, 8]
