import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.feature import corner_harris, corner_peaks
from scipy.ndimage import maximum_filter, generate_binary_structure
import tqdm
import pickle

with open('img1_features.pkl', 'rb') as file:
        img1_features = pickle.load(file)

with open('img2_features.pkl', 'rb') as file:
        img2_features = pickle.load(file)

with open('img1_pts.pkl', 'rb') as file:
        img1_pts = pickle.load(file)
        

with open('img2_pts.pkl', 'rb') as file:
        img2_pts = pickle.load(file)

def match_features(img1_features, img2_features, img1_pts, img2_pts, ratio_threshold=0.75):
    """
    Match features between two images with compatibility for tuple keypoints
    
    Args:
        img1_features: Features from first image (numpy array)
        img2_features: Features from second image (numpy array)
        img1_pts: List of (x,y) tuples from first image
        img2_pts: List of (x,y) tuples from second image
        ratio_threshold: Lowe's ratio test threshold (default: 0.75)
    
    Returns:
        matched_features: List of matched point pairs
    """
    matched_features = []
    
    # Normalize features if they aren't already
    img1_features = img1_features / (np.linalg.norm(img1_features, axis=1)[:, np.newaxis] + 1e-8)
    img2_features = img2_features / (np.linalg.norm(img2_features, axis=1)[:, np.newaxis] + 1e-8)
    
    for i, feature1 in enumerate(img1_features):
        # Compute distances to all features in img2
        distances = np.sum((img2_features - feature1) ** 2, axis=1)
        
        # Find two closest matches
        idx = np.argsort(distances)
        best_idx, second_best_idx = idx[0], idx[1]
        
        # Apply ratio test
        ratio = distances[best_idx] / (distances[second_best_idx] + 1e-8)
        
        if ratio < ratio_threshold:
            # Store matches as pairs of points
            matched_features.append([
                list(img1_pts[i]),     # Convert tuple to list
                list(img2_pts[best_idx])  # Convert tuple to list
            ])
    
    return matched_features


#print(len(img1_features))
ratio_threshold = 0.9
matched_features=[]
#lowest = np.inf

img1 = cv2.imread('./Phase1/Data/Train/Set1/1.jpg')
img2 = cv2.imread('./Phase1/Data/Train/Set1/2.jpg')


""" for i, feature1 in enumerate(img1_features):
    differences = []
    for feature2 in img2_features:
            diff = np.sum((feature1-feature2)**2)
            differences.append(diff)
    
    minima = np.argmin(differences)
    print(minima)
    best_diff = np.sort(differences)
    first,second = best_diff[:2]

    ratio = first/second

    if ratio < ratio_threshold:
            matched_features.append([img1_pts[i].tolist(),img2_pts[minima].tolist()]) """



#print(f'{len(matched_features)}No of matches')
matched_features = match_features(img1_features,img2_features,img1_pts,img2_pts)

# Convert your matched points into cv2.KeyPoint format
img1_kp = [cv2.KeyPoint(x=float(pt[0][0]), y=float(pt[0][1]), size=1.0) for pt in matched_features]
img2_kp = [cv2.KeyPoint(x=float(pt[1][0]), y=float(pt[1][1]), size=1.0) for pt in matched_features]
    
    # Create DMatch objects
matches = [cv2.DMatch(i, i, 0) for i in range(len(matched_features))]
    
    # Draw matches
matched_image = cv2.drawMatches(
        img1, img1_kp,
        img2, img2_kp,
        matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

# Show the result
cv2.imshow("Matches", matched_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
                