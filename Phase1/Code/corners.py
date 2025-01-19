import cv2
import numpy as np
import pickle

def detect_harris_corners(image, threshold=0.001):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Apply the Harris corner detector
    corner_response = cv2.cornerHarris(gray, blockSize=2, ksize=1, k=0.03)

    # Threshold for detecting strong corners
    corners = np.argwhere(corner_response > threshold * corner_response.max())

    return corners, corner_response

def adaptive_non_maximal_suppression(corners, corner_response, num_best=500):
    # Extract corner responses at corner locations
    corner_scores = [corner_response[y, x] for y, x in corners]

    # Compute suppression radius for each corner
    radii = []
    for i, (x1, y1) in enumerate(corners):
        r_min = float('inf')
        for j, (x2, y2) in enumerate(corners):
            if corner_scores[j] > corner_scores[i]:
                dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                r_min = min(r_min, dist)
        radii.append(r_min)

    # Sort corners by suppression radius in descending order
    radii = np.array(radii)
    sorted_indices = np.argsort(-radii)
    best_corners = corners[sorted_indices[:num_best]]

    return best_corners

def feature_ext(image,point):
    half_size = 41//2
   # Ensure that the extracted ROI stays within image boundaries
    height, width, _ = image.shape
    x_start = max(point[0] - half_size, 0)
    x_end = min(point[0] + half_size + 1, width)
    y_start = max(point[1] - half_size, 0)
    y_end = min(point[1]+ half_size + 1, height)

    # Extract the ROI from the image (with boundary checks)
    roi = image[y_start:y_end, x_start:x_end]

    # Apply Gaussian blur to the ROI
    blurred_roi = cv2.GaussianBlur(roi, (5, 5), 0)

    # Display the image with the blurred region (optional)
    cv2.imshow('Blurred ROI', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return blurred_roi


def main():
    # Read input image
    image = cv2.imread('./Phase1/Data/Train/Set1/2.jpg')

    # Step 1: Detect corners (Harris or Shi-Tomasi)
    corners, corner_response = detect_harris_corners(image)

    # Step 2: Apply ANMS to select evenly distributed corners
    best_corners = adaptive_non_maximal_suppression(corners, corner_response, num_best=500) 

    # Visualize the result
    for x, y in best_corners:
        cv2.circle(image, (y, x), 3, (0, 255, 0), -1)

    cv2.imshow('Corners with ANMS', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    with open('anms_output_img2.pkl', 'wb') as file:
        pickle.dump(best_corners, file)


if __name__ == "__main__":
    main()