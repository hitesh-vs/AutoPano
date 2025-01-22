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


def RANSAC(matches, points1, points2, num_iterations, threshold, inlier_percentage=0.75):
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
        H = cv2.findHomography(src_points, dst_points, method=0)[0]
        
        if H is None:
            continue
            
        # 3. Compute inliers using SSD
        current_inliers = []
        
        # Transform all points using homography
        # Reshape points to match the expected input format (N,1,2)
        src_points_reshaped = src_points_all.reshape(-1, 1, 2)
        try:
            src_points_transformed = cv2.perspectiveTransform(src_points_reshaped, H)
            
            # Compute SSD for all points
            ssd = np.sum((src_points_transformed - dst_points_all.reshape(-1, 1, 2)) ** 2, axis=(1, 2))
            
            # Find indices where SSD is below threshold
            current_inliers = np.where(ssd < threshold)[0]
            
        except cv2.error:
            continue
        
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
        best_H = cv2.findHomography(src_points, dst_points, method=0)[0]
        if best_H is not None:
            best_Hmatrix = best_H
    
    if best_Hmatrix is None:
        raise ValueError("Could not find a valid homography matrix")
        
    print(f"RANSAC completed with {len(best_inliers_set)} inliers out of {num_matches} matches")
    return best_Hmatrix, best_inliers_set

def match_images(img1, img2, ratio_thresh=0.75, save_visualization=True):
    # Process both images
    corners1, desc1 = process_single_image(img1)
    corners2, desc2 = process_single_image(img2)
    
    print(f"Number of corners detected - Image 1: {len(corners1)}, Image 2: {len(corners2)}")
    
    # Match features
    matches = match_features(desc1, desc2, ratio_thresh)
    print(f"Number of matches found: {len(matches)}")
    
    # Convert keypoints for visualization
    cv_kp1 = [cv2.KeyPoint(x=float(corner[0]), y=float(corner[1]), size=1.0) for corner in corners1]
    cv_kp2 = [cv2.KeyPoint(x=float(corner[0]), y=float(corner[1]), size=1.0) for corner in corners2]
    
    # Draw initial matches
    match_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, matches, None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    # Apply RANSAC
    try:
        H_matrix, inliers = RANSAC(matches, corners1, corners2, 1000, 5)
        
        # Filter matches using RANSAC inliers
        good_matches = [matches[idx] for idx in inliers]
        
        # Draw matches (only inliers)
        ransac_matches_img = cv2.drawMatches(img1, cv_kp1, img2, cv_kp2, good_matches, None,
                                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Get matched point pairs
        matched_pts1 = np.float32([corners1[m.queryIdx] for m in matches])
        matched_pts2 = np.float32([corners2[m.trainIdx] for m in matches])
        
    except Exception as e:
        print(f"Error in RANSAC: {str(e)}")
        raise
    
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

def find_connected_components(matches_graph):
    """
    Find groups of images that should be stitched together based on matching features.
    
    Args:
        matches_graph: Dictionary with (img_idx1, img_idx2) tuples as keys and match data as values
    
    Returns:
        List of lists, where each inner list contains indices of connected images
    """
    # Create adjacency list
    adj_list = {}
    for (img1, img2), match_data in matches_graph.items():
        # Check number of matches from the match_data dictionary
        num_matches = match_data['num_matches']
        if num_matches > 10:  # Threshold for considering images connected
            if img1 not in adj_list: adj_list[img1] = set()
            if img2 not in adj_list: adj_list[img2] = set()
            adj_list[img1].add(img2)
            adj_list[img2].add(img1)
    
    # Find connected components using DFS
    def dfs(node, component):
        component.add(node)
        for neighbor in adj_list.get(node, []):
            if neighbor not in component:
                dfs(neighbor, component)
    
    components = []
    visited = set()
    
    for node in adj_list:
        if node not in visited:
            component = set()
            dfs(node, component)
            components.append(list(component))
            visited.update(component)
    
    return components

def compute_matches_graph(images):
    """
    Compute all pairwise matches between images.
    
    Returns:
        Dictionary with (img_idx1, img_idx2) as keys and match data as values
    """
    n = len(images)
    matches_graph = {}
    
    for i in range(n):
        for j in range(i+1, n):
            try:
                results = match_images(images[i], images[j])
                matches_graph[(i, j)] = {
                    'matches': results['matches'],
                    'H_matrix': results['H_matrix'],
                    'num_matches': len(results['inliers'])
                }
                print(f"Matches between images {i} and {j}: {len(results['inliers'])}")
            except Exception as e:
                print(f"Failed to match images {i} and {j}: {str(e)}")
                matches_graph[(i, j)] = {
                    'matches': [],
                    'H_matrix': None,
                    'num_matches': 0
                }
    
    return matches_graph

def find_best_reference_image(component, matches_graph):
    """
    Find the best reference image in a component based on number of matches.
    """
    match_counts = {}
    for i in component:
        count = 0
        for j in component:
            if i == j: continue
            pair = tuple(sorted([i, j]))
            count += matches_graph[pair]['num_matches'] if pair in matches_graph else 0
        match_counts[i] = count
    
    return max(match_counts.items(), key=lambda x: x[1])[0]

def estimate_canvas_size(images, homographies, ref_idx):
    """
    Estimate the size of the final panorama canvas.
    """
    h_max, w_max = images[ref_idx].shape[:2]
    corners = np.array([[0, 0, 1],
                       [0, h_max, 1],
                       [w_max, h_max, 1],
                       [w_max, 0, 1]])
    
    for H in homographies:
        if H is not None:
            warped_corners = H.dot(corners.T).T
            warped_corners /= warped_corners[:, 2:]
            min_x = min(0, np.min(warped_corners[:, 0]))
            min_y = min(0, np.min(warped_corners[:, 1]))
            max_x = max(w_max, np.max(warped_corners[:, 0]))
            max_y = max(h_max, np.max(warped_corners[:, 1]))
            w_max = int(max_x - min_x)
            h_max = int(max_y - min_y)
    
    return h_max, w_max

def stitch_component(images, component, matches_graph):
    """
    Stitch a connected component of images together.
    """
    if len(component) == 1:
        return images[component[0]]
    
    # Find reference image
    ref_idx = find_best_reference_image(component, matches_graph)
    
    # Compute homographies relative to reference image
    homographies = [None] * len(images)
    homographies[ref_idx] = np.eye(3)
    
    def compute_homography_to_ref(idx, visited):
        if idx in visited:
            return
        visited.add(idx)
        
        # Look for direct transformation to reference
        pair = tuple(sorted([ref_idx, idx]))
        if pair in matches_graph and matches_graph[pair]['num_matches'] > 0:
            if pair[0] == ref_idx:
                homographies[idx] = matches_graph[pair]['H_matrix']
            else:
                homographies[idx] = np.linalg.inv(matches_graph[pair]['H_matrix'])
            return
        
        # Look for path through other images
        for other in component:
            if other == idx:
                continue
            pair = tuple(sorted([other, idx]))
            if pair in matches_graph and matches_graph[pair]['num_matches'] > 0:
                compute_homography_to_ref(other, visited)
                if homographies[other] is not None:
                    H_to_other = matches_graph[pair]['H_matrix'] if pair[0] == idx else np.linalg.inv(matches_graph[pair]['H_matrix'])
                    homographies[idx] = homographies[other] @ H_to_other
                    return
        """ if idx in visited:
            return
        visited.add(idx)
    
        if homographies[idx] is None:
            # Look for direct transformation to the reference
            for other in component:
                if other == idx or homographies[other] is None:
                    continue
                pair = tuple(sorted([other, idx]))
                if pair in matches_graph and matches_graph[pair]['num_matches'] > 0:
                    H_to_other = matches_graph[pair]['H_matrix'] if pair[0] == idx else np.linalg.inv(matches_graph[pair]['H_matrix'])
                    homographies[idx] = homographies[other] @ H_to_other
                    return """
    
    # Compute all homographies
    visited = set()
    for idx in component:
        compute_homography_to_ref(idx, visited)
    
    # Estimate canvas size
    h_max, w_max = estimate_canvas_size(images, [homographies[i] for i in component], ref_idx)
    
    # Create panorama
    panorama = np.zeros((h_max, w_max, 3), dtype=np.uint8)
    
    # Function to blend overlapping regions
    def blend_images(img1, img2, mask1, mask2):
        overlap = mask1 & mask2
        if not np.any(overlap):
            return img1
        
        result = img1.copy()
        overlap_coords = np.where(overlap)
        result[overlap_coords] = (img1[overlap_coords] * 0.5 + img2[overlap_coords] * 0.5).astype(np.uint8)
        non_overlap_coords = np.where(mask2 & ~mask1)
        result[non_overlap_coords] = img2[non_overlap_coords]
        return result
    
    # Initialize mask for tracking filled pixels
    panorama_mask = np.zeros((h_max, w_max), dtype=bool)
    
    # Warp and blend all images
    for idx in component:
        if homographies[idx] is None:
            continue
            
        # Warp image
        warped = cv2.warpPerspective(images[idx], homographies[idx], (w_max, h_max))
        warped_mask = warped.sum(axis=2) > 0
        
        # Blend with existing panorama
        if not np.any(panorama_mask):
            panorama = warped
            panorama_mask = warped_mask
        else:
            panorama = blend_images(panorama, warped, panorama_mask, warped_mask)
            panorama_mask |= warped_mask
    
    return panorama

def stitch_multiple_images(image_paths):
    """
    Main function to stitch multiple images into panorama(s).
    
    Args:
        image_paths: List of paths to images
    
    Returns:
        List of panorama images (one for each connected component)
    """
    # Read all images
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        images.append(img)
    
    # Compute all pairwise matches
    matches_graph = compute_matches_graph(images)
    
    # Find connected components
    components = find_connected_components(matches_graph)
    
    # Stitch each component
    panoramas = []
    for component in components:
        try:
            panorama = stitch_component(images, component, matches_graph)
            panoramas.append(panorama)
        except Exception as e:
            print(f"Failed to stitch component {component}: {str(e)}")
    
    return panoramas


if __name__ == "__main__":
    image_paths = [
        'Phase1/Data/Train/Set1/1.jpg',
        'Phase1/Data/Train/Set1/2.jpg',
        'Phase1/Data/Train/Set1/3.jpg',
        # Add more image paths as needed
    ]
    
    try:
        panoramas = stitch_multiple_images(image_paths)
        
        # Save or display results
        for i, panorama in enumerate(panoramas):
            cv2.imshow(f'Panorama {i+1}', panorama)
            cv2.imwrite(f'panorama_{i+1}.jpg', panorama)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error in panorama stitching: {str(e)}")