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

#print(len(img1_features))
ratio_threshold = 0
matched_features=[]


for feature1 in img1_features:
    differences = []
    for feature2 in img2_features:
            diff = np.sum((feature1-feature2)**2)
            differences.append(diff)

    best_diff = np.sort(differences)
    first,second = best_diff[:2]

    ratio = first/second

    if ratio < ratio_threshold:
            matched_features.append(feature1,feature2)
                