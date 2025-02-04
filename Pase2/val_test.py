# #!/usr/bin/env python
# """
# RBE/CS Fall 2022: Classical and Deep Learning Approaches for
# Geometric Computer Vision
# Project 1: MyAutoPano - Phase 2 Test Script (Multi-Sample Version)
# """

# import os
# import sys
# import glob
# import random
# import numpy as np
# import torch
# import cv2
# import argparse
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from Network.Network import HomographyModel

# # Don't generate pyc codes
# sys.dont_write_bytecode = True

# def setup_paths(base_path):
#     """
#     Set up file paths for validation data
#     """
#     val_dir = os.path.join(base_path, "Val")
#     val_files = glob.glob(os.path.join(val_dir, "*.npz"))
    
#     if not val_files:
#         raise ValueError("No validation .npz files found in directory: {}".format(val_dir))
        
#     return val_files

# def load_random_file(val_files):
#     """
#     Load a random npz file containing 5 samples
#     """
#     # Select random npz file
#     npz_file = random.choice(val_files)
    
#     # Load all data from npz file
#     data = np.load(npz_file)
    
#     # Get all 5 samples
#     pa_batch = data['PA']        # Shape: (5, H, W, 3)
#     pb_batch = data['PB']        # Shape: (5, H, W, 3)
#     h4pt_batch = data['H4Pt']    # Shape: (5, 8)

#     print(f"Shape of h4pt_batch: {h4pt_batch.shape}")
    
#     return pa_batch, pb_batch, h4pt_batch

# def denormalize_h4pt(h4pt, perturbation=32):
#     """
#     Denormalize H4Pt values from [-1, 1] range back to pixel coordinates
#     """
#     return h4pt * perturbation

# def visualize_results(pa_batch, pb_batch, h4pt_pred, h4pt_true, num_samples=1):
#     """
#     Visualize input patches and homography results for all samples
#     Args:
#         pa_batch: Array of shape (N, H, W, 3)
#         pb_batch: Array of shape (N, H, W, 3)
#         h4pt_pred: Array of shape (N, 8) or (8,)
#         h4pt_true: Array of shape (N, 8) or (8,)
#         num_samples: Number of samples to visualize
#     """
#     plt.figure(figsize=(20, 4 * num_samples))
    
#     # Ensure h4pt_pred and h4pt_true are 2D
#     if h4pt_pred.ndim == 1:
#         h4pt_pred = h4pt_pred.reshape(1, -1)  # Reshape to (1, 8)
#     if h4pt_true.ndim == 1:
#         h4pt_true = h4pt_true.reshape(1, -1)  # Reshape to (1, 8)
    
#     for i in range(num_samples):
#         # Plot original patches
#         plt.subplot(num_samples, 3, i*3+1)
#         plt.imshow(pa_batch[i])
#         plt.title(f"Sample {i+1}\nPatch A")
#         plt.axis('off')
        
#         plt.subplot(num_samples, 3, i*3+2)
#         plt.imshow(pb_batch[i])
#         plt.title(f"Sample {i+1}\nPatch B")
#         plt.axis('off')
        
#         # Plot homography points
#         plt.subplot(num_samples, 3, i*3+3)
#         plt.scatter(h4pt_true[i, ::2], h4pt_true[i, 1::2], c='r', label='Ground Truth')
#         plt.scatter(h4pt_pred[i, ::2], h4pt_pred[i, 1::2], c='b', marker='x', label='Predicted')
#         plt.title(f"Sample {i+1}\nHomography Points Comparison")
#         plt.legend()
#         plt.axis('equal')
    
#     plt.tight_layout()
#     plt.show()

# def test_operation(model_path, base_path):
#     """
#     Main test operation with 6-channel input
#     """
#     # Setup device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
#     # Initialize model
#     model = HomographyModel().to(device)
    
#     # Load checkpoint
#     if not os.path.exists(model_path):
#         raise FileNotFoundError("Model checkpoint not found at: {}".format(model_path))
        
#     checkpoint = torch.load(model_path, map_location=device)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     model.eval()
    
#     # Get validation files
#     val_files = setup_paths(base_path)
    
#     # Load random file with 5 samples
#     pa_batch, pb_batch, h4pt_true = load_random_file(val_files)
    
#     # Convert to tensor and stack channels
#     pa_tensor = torch.from_numpy(pa_batch).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)  # Shape: (5, 3, H, W)
#     pb_tensor = torch.from_numpy(pb_batch).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)  # Shape: (5, 3, H, W)
    
#     # Stack along channel dimension to create 6-channel input
#     combined_input = torch.cat([pa_tensor, pb_tensor], dim=1)  # Shape: (5, 6, H, W)
    
#     # Run inference on all 5 samples
#     with torch.no_grad():
#         h4pt_pred = model(pa_tensor,pb_tensor).cpu().numpy()  # Modified to take single input
    
#     # Denormalize predictions
#     h4pt_pred_denorm = denormalize_h4pt(h4pt_pred)
#     h4pt_true_denorm = denormalize_h4pt(h4pt_true)
    
#     # Print results for first sample
#     print("\nFirst Sample Results:")
#     print("Predicted H4Pt (denormalized):")
#     print(h4pt_pred_denorm[0])
#     print("\nGround Truth H4Pt (denormalized):")
#     print(h4pt_true_denorm[0])
    
#     # Visualize all 5 samples
#     visualize_results(pa_batch, pb_batch, h4pt_pred_denorm, h4pt_true_denorm)

# def main():
#     """
#     Main function
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--model_path",
#         type=str,
#         default="Checkpoints/0a1model.ckpt",
#         help="Path to trained model checkpoint"
#     )
#     parser.add_argument(
#         "--data_path",
#         type=str,
#         default="generated_data_color_25k",
#         help="Path to dataset directory containing Val folder"
#     )
    
#     args = parser.parse_args()
    
#     test_operation(args.model_path, args.data_path)

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
"""
RBE/CS Fall 2022: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project 1: MyAutoPano - Phase 2 Test Script (Single-Sample Version)
"""

import os
import sys
import glob
import random
import numpy as np
import torch
import cv2
import argparse
import matplotlib.pyplot as plt
from Network.Network_unsupervised import HomographyModel

# Don't generate pyc codes
sys.dont_write_bytecode = True

def setup_paths(base_path):
    """
    Set up file paths for validation data
    """
    val_dir = os.path.join(base_path, "Train")
    val_files = glob.glob(os.path.join(val_dir, "*.npz"))
    
    if not val_files:
        raise ValueError("No validation .npz files found in directory: {}".format(val_dir))
        
    return val_files

def load_random_file(val_files):
    """
    Load a random npz file containing a single sample
    """
    # Select random npz file
    npz_file = random.choice(val_files)
    #npz_file = "data_25k/Train/313_sample1.npz"

    print(f"Loading file: {npz_file}")
    
    # Load data from npz file
    data = np.load(npz_file)

    # Extract the image index from the filename
    base_name = os.path.basename(npz_file)  # Example: "284_sample2.npz"
    image_index = base_name.split('_')[0]   # Extracts "284"

    image_folder = r"Phase2/Data/Train"

    # Construct the corresponding image file path
    image_path = os.path.join(image_folder, f"{image_index}.jpg")
    print(f"Loading corresponding image: {image_path}")

    # Load the image
    if os.path.exists(image_path):
        image = cv2.imread(image_path)  # Use cv2 (BGR format) or Image.open(image_path) for PIL
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct display
    else:
        print(f"Warning: Image {image_path} not found!")
        image = None  # Return None if the image is missing
    
    # Get the single sample
    pa = data['PA']        # Shape: (3, 128, 128)
    pb = data['PB']        # Shape: (3, 128, 128)
    h4pt = data['H4Pt']    # Shape: (8,)
    ca = data['CA']        # Shape: (4, 2)
    
    # Debug: Print shapes
    print(f"Shape of PA: {pa.shape}")
    print(f"Shape of PB: {pb.shape}")
    print(f"Shape of H4Pt: {h4pt.shape}")
    print(f"Shape of CA: {ca.shape}")
    
    return pa, pb, h4pt, ca, image

def denormalize_h4pt(h4pt, perturbation=32):
    """
    Denormalize H4Pt values from [-1, 1] range back to pixel coordinates
    """
    return h4pt * perturbation

def visualize_results(pa, pb, h4pt_pred, h4pt_true,ca,image):
    """
    Visualize input patches and homography results
    Args:
        pa: Array of shape (3, 128, 128)
        pb: Array of shape (3, 128, 128)
        h4pt_pred: Array of shape (8,)
        h4pt_true: Array of shape (8,)
    """
    # plt.figure(figsize=(15, 5))
    
    # # Convert CHW to HWC for visualization
    # #pa = np.transpose(pa, (0, 2, 1))  # Shape: (128, 128, 3)
    # #pb = np.transpose(pb, (0, 2, 1))  # Shape: (128, 128, 3)
    
    # # Plot original patches
    # plt.subplot(1, 3, 1)
    # plt.imshow(pa)
    # plt.title("Patch A")
    # plt.axis('off')
    
    # plt.subplot(1, 3, 2)
    # plt.imshow(pb)
    # plt.title("Patch B")
    # plt.axis('off')
    
    # # Plot homography points
    # plt.subplot(1, 3, 3)
    # plt.scatter(h4pt_true[::2], h4pt_true[1::2], c='r', label='Ground Truth')
    # plt.scatter(h4pt_pred[::2], h4pt_pred[1::2], c='b', marker='x', label='Predicted')
    # plt.title("Homography Points Comparison")
    # plt.legend()
    # plt.axis('equal')
    
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 5))

    # Convert CHW to HWC for visualization
    #pa = np.transpose(pa, (0, 2, 1))  # Shape: (128, 128, 3)
    #pb = np.transpose(pb, (0, 2, 1))  # Shape: (128, 128, 3)

    #ca[:, :, [0, 1]] = ca[:, :, [1, 0]]

    # Compute the warped corner points
    corners_gt = (ca.squeeze() + h4pt_true.reshape(4, 2))  # (4, 2)
    corners_pred = (ca.squeeze() + h4pt_pred.reshape(4, 2))  # (4, 2)\ 
    
    # corners_gt = (ca.squeeze())  # (4, 2)
    # corners_pred = (ca.squeeze())

    # Plot original patch A
    plt.subplot(1, 3, 1)
    plt.imshow(pa)
    plt.title("Patch A")
    plt.axis('off')

    # Plot patch B  
    plt.subplot(1, 3, 2)
    plt.imshow(pb)
    plt.title("Patch B")
    plt.axis('off')

    # Plot patch B with the quadrilateral overlay
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.title("Warped Patch Overlay")
    plt.axis('off')

    # Draw ground truth quadrilateral (Red)
    plt.plot(*zip(*np.vstack([corners_gt, corners_gt[0]])), 'r-', label="Ground Truth")

    # Draw predicted quadrilateral (Blue)
    plt.plot(*zip(*np.vstack([corners_pred, corners_pred[0]])), 'b--', label="Predicted")

    plt.legend()
    plt.show()

def test_operation(model_path, base_path):
    """
    Main test operation with single-sample input
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = HomographyModel().to(device)
    
    # Load checkpoint
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model checkpoint not found at: {}".format(model_path))
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get validation files
    val_files = setup_paths(base_path)
    
    # Load random file with a single sample
    pa, pb, h4pt_true,ca,image = load_random_file(val_files)

    image = cv2.resize(image, (320, 240))
    
    # Convert to tensor and add batch dimension
    pa_tensor = torch.from_numpy(pa).unsqueeze(0).float().to(device)  # Shape: (1, 3, 128, 128)
    pb_tensor = torch.from_numpy(pb).unsqueeze(0).float().to(device)  # Shape: (1, 3, 128, 128)

    pa_tensor = pa_tensor.permute(0, 3, 1, 2)
    pb_tensor = pb_tensor.permute(0, 3, 1, 2)
    
    
    # Run inference on the single sample
    with torch.no_grad():
        h4pt_pred = model(pa_tensor, pb_tensor).cpu().numpy()  # Shape: (1, 8)
    
    # Remove batch dimension
    h4pt_pred = h4pt_pred.squeeze(0)  # Shape: (8,)
    
    # Denormalize predictions
    h4pt_pred_denorm = denormalize_h4pt(h4pt_pred)
    h4pt_true_denorm = denormalize_h4pt(h4pt_true)
    
    # Print results
    print("\nResults:")
    print("Predicted H4Pt (denormalized):")
    print(h4pt_pred_denorm)
    print("\nGround Truth H4Pt (denormalized):")
    print(h4pt_true_denorm)
    
    # Visualize results
    visualize_results(pa, pb, h4pt_pred_denorm, h4pt_true_denorm,ca,image)

def main():
    """
    Main function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Checkpoints/25model.ckpt",
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data_25k",
        help="Path to dataset directory containing Val folder"
    )
    
    args = parser.parse_args()
    
    test_operation(args.model_path, args.data_path)

if __name__ == "__main__":
    main()
