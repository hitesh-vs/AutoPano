# import numpy as np
# import cv2

# def apply_anms(harris_corners, num_best=1000, min_distance=10):
#     """
#     Apply Adaptive Non-Maximal Suppression to Harris corner responses.
    
#     Args:
#         harris_corners: Harris corner response matrix
#         num_best: Number of corners to retain
#         min_distance: Minimum distance between corners
        
#     Returns:
#         List of (x, y) coordinates of selected corners
#     """
#     # Threshold and get coordinates of Harris corners
#     threshold = 0.01 * harris_corners.max()  # adjust this value as needed
#     corner_coords = np.where(harris_corners > threshold)
#     corners = np.column_stack((corner_coords[1], corner_coords[0]))  # x,y format
#     corner_strengths = harris_corners[corner_coords]
    
#     # Initialize arrays for ANMS
#     N = len(corners)
#     robustness = np.full(N, np.inf)
    
#     # For each corner, find the minimum distance to a stronger corner
#     for i in range(N):
#         for j in range(N):
#             if corner_strengths[j] > corner_strengths[i]:
#                 dist = np.linalg.norm(corners[i] - corners[j])
#                 robustness[i] = min(robustness[i], dist)
    
#     # Sort corners by robustness
#     indices = np.argsort(-robustness)  # Descending order
#     selected_corners = corners[indices[:num_best]]
    
#     # Apply minimum distance constraint
#     final_corners = []
#     for corner in selected_corners:
#         if not final_corners or all(np.linalg.norm(corner - fc) >= min_distance for fc in final_corners):
#             final_corners.append(corner)
    
#     return np.array(final_corners)

# def visualize_corners(image, corners):
#     """
#     Visualize detected corners on the image.
    
#     Args:
#         image: Original image
#         corners: Array of (x, y) corner coordinates
#     """
#     vis_image = image.copy()
#     for x, y in corners:
#         cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 0), -1)
#     return vis_image

# # Example usage:
# def detect_and_suppress_corners(image, harris_params={'blockSize': 3, 'ksize': 3, 'k': 0.03}, 
#                               anms_params={'num_best': 1000, 'min_distance': 5}):
#     """
#     Complete pipeline for corner detection with ANMS.
#     """
#     # Convert to grayscale and float32
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     gray = np.float32(gray)
    
#     # Apply Gaussian blur
#     gray = cv2.GaussianBlur(gray, (3,3), 0)
    
#     # Detect Harris corners
#     harris_corners = cv2.cornerHarris(gray, **harris_params) # returns corner response matrix
    
#     # Apply ANMS
#     selected_corners = apply_anms(harris_corners, **anms_params)
    
#     # Visualize results
#     result = visualize_corners(image, selected_corners)
    
#     return result, selected_corners


# def main():
#     # Load image
#     image = cv2.imread(r'C:\Users\farha\Downloads\AutoPano-new\AutoPano-new\Data\1.jpg')
    
#     # Detect and suppress corners
#     result, corners = detect_and_suppress_corners(image)
    
#     # Display the result
#     cv2.imshow('Corners', result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     main()


# import numpy as np
# import cv2

# def detect_harris_corners(image, block_size=3, ksize=3, k=0.04):
#     """
#     Detect Harris corners in the image.
#     """
#     # Convert to grayscale and float32
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image.copy()
#     gray = np.float32(gray)
    
#     # Apply Gaussian blur
#     gray = cv2.GaussianBlur(gray, (3,3), 0)
    
#     # Detect Harris corners
#     harris_corners = cv2.cornerHarris(gray, block_size, ksize, k)
#     print(len(harris_corners))
    
#     return harris_corners

# def apply_anms(harris_corners, num_best=1000, min_distance=7):
#     """
#     Apply Adaptive Non-Maximal Suppression to Harris corner responses.
#     """
#     # Threshold and get coordinates of Harris corners
#     threshold = 0.005 * harris_corners.max()
#     corner_coords = np.where(harris_corners > threshold)
#     corners = np.column_stack((corner_coords[1], corner_coords[0]))  # x,y format
#     corner_strengths = harris_corners[corner_coords]
    
#     # Initialize arrays for ANMS
#     N = len(corners)
#     robustness = np.full(N, np.inf)
    
#     # For each corner, find the minimum distance to a stronger corner
#     for i in range(N):
#         for j in range(N):
#             if corner_strengths[j] > corner_strengths[i]:
#                 dist = np.linalg.norm(corners[i] - corners[j])
#                 robustness[i] = min(robustness[i], dist)
    
#     # Sort corners by robustness
#     indices = np.argsort(-robustness)  # Descending order
#     selected_corners = corners[indices[:num_best]]
    
#     # Apply minimum distance constraint
#     final_corners = []
#     for corner in selected_corners:
#         if not final_corners or all(np.linalg.norm(corner - fc) >= min_distance for fc in final_corners):
#             final_corners.append(corner)
    
#     return np.array(final_corners)

# def get_feature_descriptors(image, corner_points):
#     """
#     Generate feature descriptors for given corner points.
#     """
#     # Convert to grayscale if needed
#     if len(image.shape) == 3:
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     else:
#         gray = image.copy()
    
#     # Parameters
#     patch_size = 41
#     output_size = 8
#     half_patch = patch_size // 2
    
#     # Initialize output array
#     num_corners = len(corner_points)
#     feature_vectors = np.zeros((num_corners, output_size * output_size))
    
#     # Process each corner point
#     for idx, corner in enumerate(corner_points):
#         x, y = int(corner[0]), int(corner[1])
        
#         # Handle boundary cases
#         if (x - half_patch < 0 or x + half_patch >= image.shape[1] or 
#             y - half_patch < 0 or y + half_patch >= image.shape[0]):
#             pad_width = ((half_patch, half_patch), (half_patch, half_patch))
#             padded_image = np.pad(gray, pad_width, mode='reflect')
#             # Adjust coordinates for padded image
#             patch = padded_image[
#                 y:y + patch_size,
#                 x:x + patch_size
#             ]
#         else:
#             # Extract patch (41x41) centered at corner point
#             patch = gray[
#                 y - half_patch:y + half_patch + 1,
#                 x - half_patch:x + half_patch + 1
#             ]
        
#         # Apply Gaussian blur
#         blurred_patch = cv2.GaussianBlur(patch, (3, 3), sigmaX=1.5)
        
#         # Subsample to 8x8
#         resized_patch = cv2.resize(blurred_patch, (output_size, output_size))
        
#         # Reshape to 64x1 vector
#         feature_vector = resized_patch.reshape(-1)
        
#         # Standardize to zero mean and unit variance
#         if feature_vector.std() != 0:
#             feature_vector = (feature_vector - feature_vector.mean()) / feature_vector.std()
        
#         feature_vectors[idx] = feature_vector
    
#     return feature_vectors

# def visualize_results(image, corners, save_path=None):
#     """
#     Visualize detected corners on the image.
#     """
#     vis_image = image.copy()
#     for x, y in corners:
#         cv2.circle(vis_image, (int(x), int(y)), 3, (0, 255, 0), -1)
    
#     if save_path:
#         cv2.imwrite(save_path, vis_image)
    
#     return vis_image

# def process_image(image_path, save_visualization=True):
#     """
#     Complete pipeline for processing an image:
#     1. Detect Harris corners
#     2. Apply ANMS
#     3. Generate feature descriptors
#     """
#     # Read image
#     image = cv2.imread(image_path)
#     if image is None:
#         raise ValueError(f"Could not read image at {image_path}")
    
#     # Detect Harris corners
#     harris_corners = detect_harris_corners(
#         image,
#         block_size=2,
#         ksize=3,
#         k=0.02
#     )
    
#     # Apply ANMS
#     corner_points = apply_anms(
#         harris_corners,
#         num_best=1000,
#         min_distance=7
#     )
    
#     # Generate feature descriptors
#     descriptors = get_feature_descriptors(image, corner_points)
    
#     # Visualize results
#     if save_visualization:
#         result_image = visualize_results(
#             image, 
#             corner_points,
#             save_path='corners_detected.jpg'
#         )
#     else:
#         result_image = visualize_results(image, corner_points)
    
#     return {
#         'corners': corner_points,
#         'descriptors': descriptors,
#         'visualization': result_image
#     }

# # Example usage
# if __name__ == "__main__":
#     # Process single image
#     image1_path = r'C:\Users\farha\Downloads\AutoPano-new\AutoPano-new\Data\1.jpg'  # Replace with your image path
#     image2_path = r'C:\Users\farha\Downloads\AutoPano-new\AutoPano-new\Data\2.jpg'  # Replace with your image path

#     try:
#         results1 = process_image(image1_path)
#         results2 = process_image(image2_path)
        
#         # Print information about results
#         print(f"Number of corners in image1 detected: {len(results1['corners'])}")
#         print(f"Feature descriptor image 1 shape: {results1['descriptors'].shape}")

#         print(f"Number of corners in image2 detected: {len(results2['corners'])}")
#         print(f"Feature descriptor image 2 shape: {results2['descriptors'].shape}") 

#         # Display results
#         cv2.imshow('Detected Corners', results1['visualization'])
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#                 # Display results
#         cv2.imshow('Detected Corners', results2['visualization'])
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()
        
#     except Exception as e:
#         print(f"Error processing image: {str(e)}")


import numpy as np
import cv2
import os

def detect_harris_corners(image, block_size=2, ksize=3, k=0.02):

    # Convert to grayscale and float32
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = np.float32(gray)
    
    # Apply Gaussian blur
    gray = cv2.GaussianBlur(gray, (3,3), 0)
    
    # Detect Harris corners
    harris_corners = cv2.cornerHarris(gray, block_size, ksize, k)
    
    return harris_corners

def apply_anms(harris_corners, num_best=1000, min_distance=7):
    """
    Apply Adaptive Non-Maximal Suppression to Harris corner responses.
    """
    # Threshold and get coordinates of Harris corners
    threshold = 0.005 * harris_corners.max()
    corner_coords = np.where(harris_corners > threshold)
    corners = np.column_stack((corner_coords[1], corner_coords[0]))  # x,y format
    corner_strengths = harris_corners[corner_coords]
    
    # Initialize arrays for ANMS
    N = len(corners)
    robustness = np.full(N, np.inf)
    
    # For each corner, find the minimum distance to a stronger corner
    for i in range(N):
        for j in range(N):
            if corner_strengths[j] > corner_strengths[i]:
                dist = np.linalg.norm(corners[i] - corners[j])
                robustness[i] = min(robustness[i], dist)
    
    # Sort corners by robustness
    indices = np.argsort(-robustness)  # Descending order
    selected_corners = corners[indices[:num_best]]
    
    # Apply minimum distance constraint -- we can remove this if not needed
    # final_corners = []
    # for corner in selected_corners:
    #     if not final_corners or all(np.linalg.norm(corner - fc) >= min_distance for fc in final_corners):
    #         final_corners.append(corner)
    
    return np.array(selected_corners)

def get_feature_descriptors(image, corner_points):

    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Parameters
    patch_size = 41
    output_size = 8
    half_patch = patch_size // 2
    
    # Initialize output array
    num_corners = len(corner_points)
    feature_vectors = np.zeros((num_corners, output_size * output_size))
    
    # Process each corner point
    for idx, corner in enumerate(corner_points):
        x, y = int(corner[0]), int(corner[1])
        
        # Handle boundary cases
        if (x - half_patch < 0 or x + half_patch >= image.shape[1] or 
            y - half_patch < 0 or y + half_patch >= image.shape[0]):
            pad_width = ((half_patch, half_patch), (half_patch, half_patch))
            padded_image = np.pad(gray, pad_width, mode='reflect')
            # Adjust coordinates for padded image
            patch = padded_image[
                y:y + patch_size,   # y - half_patch + half_patch(padding offset) : y + half_patch + half_patch(padding offset) + 1
                x:x + patch_size
            ]
        else:
            # Extract patch (41x41) centered at corner point
            patch = gray[
                y - half_patch:y + half_patch + 1,
                x - half_patch:x + half_patch + 1
            ]
        
        # Apply Gaussian blur
        blurred_patch = cv2.GaussianBlur(patch, (3, 3), sigmaX=1.5)
        
        # Subsample to 8x8
        resized_patch = cv2.resize(blurred_patch, (output_size, output_size))
        
        # Reshape to 64x1 vector
        feature_vector = resized_patch.reshape(-1)
        
        # Standardize to zero mean and unit variance
        if feature_vector.std() != 0:
            feature_vector = (feature_vector - feature_vector.mean()) / feature_vector.std()
        
        feature_vectors[idx] = feature_vector
    
    return feature_vectors

def match_features(desc1, desc2, ratio_thresh=0.75):

    matches = []
    
    # For each descriptor in first image
    for idx1, desc1_single in enumerate(desc1):
        # Compute distances to all descriptors in second image
        distances = np.sqrt(np.sum((desc2 - desc1_single)**2, axis=1))
        
        # Sort distances and get indices
        sorted_idx = np.argsort(distances)
        
        # Get best and second best matches
        best_idx = sorted_idx[0]
        second_best_idx = sorted_idx[1]
        
        # Apply ratio test
        if distances[best_idx] < ratio_thresh * distances[second_best_idx]:
            match = cv2.DMatch(idx1, best_idx, distances[best_idx])
            matches.append(match)
    
    return matches

def process_single_image(image):

    harris_corners = detect_harris_corners(image)
    corners = apply_anms(harris_corners)
    descriptors = get_feature_descriptors(image, corners)
    
    return corners, descriptors


def match_images(img1, img2, ratio_thresh=0.75, save_visualization=True):

    # Process both images
    corners1, desc1 = process_single_image(img1)
    corners2, desc2 = process_single_image(img2)
    
    print(f"Number of corners detected - Image 1: {len(corners1)}, Image 2: {len(corners2)}")
    
    # Match features
    matches = match_features(desc1, desc2, ratio_thresh)
    print(f"Number of matches found: {len(matches)}")
    
    # Convert keypoints for visualization - with explicit type conversion and error checking
    cv_kp1 = []
    cv_kp2 = []
    
    # Debug print to check corner format
    print(f"Sample corner1 format: {corners1[0] if len(corners1) > 0 else 'No corners'}")
    
    try:
        for corner in corners1:
            x, y = float(corner[0]), float(corner[1])
            cv_kp1.append(cv2.KeyPoint(x=x, y=y, size=1.0))
            
        for corner in corners2:
            x, y = float(corner[0]), float(corner[1])
            cv_kp2.append(cv2.KeyPoint(x=x, y=y, size=1.0))
    except Exception as e:
        print(f"Error converting corners to KeyPoints: {str(e)}")
        print(f"Corner data type: {type(corners1[0])}")
        print(f"Corner values: {corners1[0]}")
        raise
    
    # Draw matches
    match_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    

    H_matrix, inliers = RANSAC(matches,corners1,corners2, 1000, 5)
    
    # Filter matches using RANSAC inliers
    good_matches = [matches[idx] for idx in inliers]
    
    # Draw matches (only inliers)
    ransac_matches_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, good_matches, None,
                                       flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    # if save_visualization:
    #     cv2.imwrite('matched_features_13.jpg', match_img)
    #     cv2.imwrite('ransac_matches_13.jpg', ransac_matches_img)
    
    # Get matched point pairs
    if len(matches) > 0:
        matched_pts1 = np.float32([corners1[m.queryIdx] for m in matches])
        matched_pts2 = np.float32([corners2[m.trainIdx] for m in matches])
    else:
        matched_pts1, matched_pts2 = np.array([]), np.array([])
    

   

    return {
        'matches': matches,
        'matched_points1': matched_pts1,
        'matched_points2': matched_pts2,
        'feature_matches': match_img,
        'corners1': corners1,
        'corners2': corners2,
        'descriptors1': desc1,
        'descriptors2': desc2,
        'H_matrix': H_matrix,
        'inliers': inliers,
        'ransac_matches': ransac_matches_img
    }


def RANSAC(matches,points1,points2,num_iterations, threshold,inlier_percentage=0.75):

    num_matches = len(matches)
    best_inliers_set = []
    best_Hmatrix = None

      # Convert matches to numpy arrays first
    src_points_all = np.float32([points1[m.queryIdx] for m in matches])
    dst_points_all = np.float32([points2[m.trainIdx] for m in matches])
    
    for iteration in range(num_iterations):
        # 1. Randomly select 4 feature pairs
        
        random_indices = np.random.choice(num_matches, 4, replace=False)
        
        # Get the corresponding points
        src_points = src_points_all[random_indices]
        dst_points = dst_points_all[random_indices]
        
        # 2. Compute homography H for these points

        H = cv2.findHomography(src_points.reshape(-1,1,2), 
                                dst_points.reshape(-1,1,2), 
                            method=0)[0]
        
        
            
        # 3. Compute inliers using SSD
        current_inliers = []
        
        # Transform all points at once
        src_points_transformed = cv2.perspectiveTransform(
            src_points_all.reshape(-1,1,2), H
        )
        
        # Compute SSD for all points at once
        ssd = np.sum((src_points_transformed - dst_points_all.reshape(-1,1,2)) ** 2, axis=(1,2))
        
        # Find indices where SSD is below threshold
        current_inliers = np.where(ssd < threshold)[0]
        
        # 4. Check if this is the best set of inliers so far
        if len(current_inliers) > len(best_inliers_set):
            best_inliers_set = current_inliers
            best_Hmatrix = H
        
        # Check if we've found enough inliers
        inlier_ratio = len(current_inliers) / num_matches
        if inlier_ratio > inlier_percentage:
            break
                
    
    # 6. Re-compute H using all inliers
    if len(best_inliers_set) > 0:
        src_points = src_points_all[best_inliers_set]
        dst_points = dst_points_all[best_inliers_set]
        best_H = cv2.findHomography(src_points.reshape(-1,1,2), 
                                  dst_points.reshape(-1,1,2), 
                                  method=0)[0]
    
    print(f"RANSAC completed with {len(best_inliers_set)} inliers out of {num_matches} matches")
    return best_H, best_inliers_set

def stitch_images(img1, img2, H_matrix):
    """
    Stitch two images together using a homography matrix.
    
    Args:
        img1: Source image to be warped
        img2: Target image
        H_matrix: Homography matrix mapping img1 to img2's perspective
    
    Returns:
        Stitched panorama image
    """
    # Get dimensions of both images
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Create points for corners of img1
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    
    # Transform corners of img1
    corners1_transformed = cv2.perspectiveTransform(corners1, H_matrix)
    
    # Get the minimum x and y coordinates
    min_x = int(min(min(corners1_transformed[:, 0, 0]), 0))
    min_y = int(min(min(corners1_transformed[:, 0, 1]), 0))
    
    # Get the maximum x and y coordinates
    max_x = int(max(max(corners1_transformed[:, 0, 0]), w2))
    max_y = int(max(max(corners1_transformed[:, 0, 1]), h2))
    
    # Calculate offset for translation
    offset_x = -min_x
    offset_y = -min_y
    
    # Create translation matrix
    translation_matrix = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ])
    
    # Combine translation with homography
    final_matrix = translation_matrix.dot(H_matrix)
    
    # Calculate size of panorama
    output_width = max_x + offset_x
    output_height = max_y + offset_y
    
    # Warp first image to create panorama
    warped_img = cv2.warpPerspective(img1, final_matrix, (output_width, output_height))
    
    # Create final output image
    output_img = warped_img.copy()
    
    # Paste second image into the output image at the correct position
    output_img[offset_y:offset_y+h2, offset_x:offset_x+w2] = img2
    
    return output_img


if __name__ == "__main__":
    folder_path = r'Phase1/Data/Train/Set1/'  
    output_folder = r'panorama_results/'  

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get list of images in folder
    image_files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    if len(image_files) < 2:
        raise ValueError("Need at least two images in the folder to stitch a panorama")

    # Initialize panorama with the first image
    panorama = cv2.imread(image_files[0])
    if panorama is None:
        raise ValueError(f"Could not read the initial image: {image_files[0]}")

    # Process each subsequent image
    for i, image_path in enumerate(image_files[1:], start=1):
        try:
            img2 = cv2.imread(image_path)
            if img2 is None:
                print(f"Could not read image: {image_path}")
                continue

            # Match and stitch images
            results = match_images(panorama, img2)
            H_matrix = results['H_matrix']

            panorama = stitch_images(panorama, img2, H_matrix)

            # Save intermediate panorama
            output_file = os.path.join(output_folder, f'panorama_part{i}.jpg')
            cv2.imwrite(output_file, panorama)
            print(f"Saved panorama: {output_file}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")

    # Save final panorama
    final_output = os.path.join(output_folder, 'final_panorama.jpg')
    cv2.imwrite(final_output, panorama)
    print(f"Final panorama saved as: {final_output}")
