""" import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.feature import corner_harris, corner_peaks
from scipy.ndimage import maximum_filter, generate_binary_structure
import tqdm
import pickle

def process_patches(image, keypoints, patch_size=41, output_size=8):
    
    half_patch = patch_size // 2
    valid_keypoints = []
    feature_vectors = []

    for keypoint in keypoints:
        x, y = keypoint
        
        # Check if the patch around the keypoint is fully contained in the image
        if (x - half_patch >= 0 and x + half_patch < image.shape[1] and
            y - half_patch >= 0 and y + half_patch < image.shape[0]):
            
            # Extract the 41×41 patch
            patch = image[x - half_patch:x + half_patch + 1, y - half_patch:y + half_patch + 1]
            
            # Apply Gaussian blur
            blurred_patch = cv2.GaussianBlur(patch, (0, 0), sigmaX=1, sigmaY=1)
            
            # Sub-sample the blurred patch to 8×8
            sub_sampled_patch = cv2.resize(blurred_patch, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
            
            # Reshape to a 64×1 vector
            feature_vector = sub_sampled_patch.flatten()
            
            # Standardize the vector
            mean = np.mean(feature_vector)
            std = np.std(feature_vector)
            standardized_vector = (feature_vector - mean) / (std + 1e-8)  # Add small epsilon for numerical stability
            
            # Append results
            valid_keypoints.append(keypoint)
            feature_vectors.append(standardized_vector)
    
    return valid_keypoints, feature_vectors

def main():

    image_path = './Phase1/Data/Train/Set1/1.jpg'

    img_test = cv2.imread(image_path)

    with open('anms_output.pkl', 'rb') as file:
        supressed = pickle.load(file)

    #print(len(supressed))

    valid_pts,features = process_patches(img_test,supressed)

    print(len(features))

    for x, y in valid_pts:
        cv2.circle(img_test, (y, x), 3, (0, 255, 0), -1)

    cv2.imshow('Corners with ANMS', img_test)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    


if __name__ =="__main__":
    main() """

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.feature import corner_harris, corner_peaks
from scipy.ndimage import maximum_filter, generate_binary_structure
import tqdm
import pickle

def process_patches(image, keypoints, patch_size=41, output_size=8):
    """
    Process patches centered around keypoints with improved feature extraction.
    
    Parameters:
        image (ndarray): Input image (grayscale)
        keypoints (list): List of (x, y) coordinates
        patch_size (int): Size of patch (must be odd)
        output_size (int): Size after sub-sampling
        
    Returns:
        valid_keypoints (list): Filtered keypoints
        feature_vectors (ndarray): Normalized feature vectors
    """
    if patch_size % 2 == 0:
        patch_size += 1  # Ensure odd size for proper centering
        
    half_patch = patch_size // 2
    valid_keypoints = []
    feature_vectors = []
    
    # Pre-compute Gaussian kernel
    gaussian_kernel = cv2.getGaussianKernel(5, 1)
    gaussian_kernel = gaussian_kernel @ gaussian_kernel.T
    
    # Add padding to image to handle border keypoints
    padded_image = cv2.copyMakeBorder(
        image,
        half_patch, half_patch, half_patch, half_patch,
        cv2.BORDER_REFLECT_101
    )
    
    for x, y in keypoints:
        try:
            # Extract patch (notice x, y are swapped for correct orientation)
            patch = padded_image[
                y:y + patch_size,
                x:x + patch_size
            ]
            
            # Apply preprocessing steps
            # 1. Gaussian smoothing for noise reduction
            smoothed = cv2.filter2D(patch, -1, gaussian_kernel)
            
            # 2. Compute gradient magnitude and orientation
            grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            orientation = np.arctan2(grad_y, grad_x)
            
            # 3. Weight gradients by magnitude
            weighted_gradients = magnitude * orientation
            
            # 4. Sub-sample with anti-aliasing
            resized = cv2.resize(
                weighted_gradients,
                (output_size, output_size),
                interpolation=cv2.INTER_AREA
            )
            
            # 5. Flatten and normalize
            feature = resized.flatten()
            feature = feature - np.mean(feature)
            norm = np.linalg.norm(feature)
            if norm > 1e-6:  # Avoid division by zero
                feature = feature / norm
                
            # 6. Apply contrast normalization
            threshold = 0.2
            feature = np.minimum(threshold, feature)
            feature = feature / np.linalg.norm(feature)
            
            valid_keypoints.append((x, y))
            feature_vectors.append(feature)
            
        except Exception as e:
            continue
            
    if not feature_vectors:
        return [], np.array([])
        
    return valid_keypoints, np.array(feature_vectors)

def main():

    image_path = './Phase1/Data/Train/Set1/2.jpg'
    
    # Load image and convert to grayscale
    img_color = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    
    # Load suppressed points
    with open('anms_output_img2.pkl', 'rb') as file:
        suppressed = pickle.load(file)
    
    # Process patches using grayscale image
    valid_pts, features = process_patches(img_gray, suppressed)
    
    print(f"Number of valid keypoints: {len(valid_pts)}")
    print(f"Number of feature vectors: {len(features)}")
    
    # Draw valid points on color image
    img_display = img_color.copy()
    for x, y in valid_pts:
        # Draw circle using consistent coordinates
        cv2.circle(img_display, (y, x), 3, (0, 255, 0), -1)
    
    # Display results
    cv2.imshow('Valid Keypoints', img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open('img2_pts.pkl', 'wb') as file:
        pickle.dump(valid_pts, file)
    
    with open('img2_features.pkl', 'wb') as file:
        pickle.dump(features, file)

if __name__ == "__main__":
    main()
