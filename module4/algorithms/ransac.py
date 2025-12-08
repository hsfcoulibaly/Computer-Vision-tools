import numpy as np


def compute_homography(p1, p2):
    """
    Computes the Homography matrix H using the Direct Linear Transform (DLT)
    method, requiring at least 4 corresponding points (p1 and p2).

    Args:
        p1, p2 (np.array): Nx2 arrays of corresponding (x, y) coordinates.

    Returns:
        np.array: The 3x3 Homography matrix H.
    """
    # Requires setting up a system of linear equations A * h = 0 (2 rows per point)
    # The solution h is the eigenvector corresponding to the smallest eigenvalue of A^T * A.

    num_points = p1.shape[0]
    A = np.zeros((2 * num_points, 9))

    for i in range(num_points):
        x, y = p1[i, 0], p1[i, 1]
        xp, yp = p2[i, 0], p2[i, 1]

        # Set up matrix A for a single point pair (p -> p')
        A[2 * i] = [-x, -y, -1, 0, 0, 0, xp * x, xp * y, xp]
        A[2 * i + 1] = [0, 0, 0, -x, -y, -1, yp * x, yp * y, yp]

    # Use SVD to solve A * h = 0 -> h is the last column of V (from U S V^T)
    _, _, V = np.linalg.svd(A)
    H = V[-1, :].reshape(3, 3)

    # Normalize H (h_33 = 1)
    H /= H[2, 2]
    return H


def ransac_homography(matches_kps1, matches_kps2, max_iter=2000, threshold=5.0, min_inliers=10):
    """
    Estimates the best Homography matrix using the RANSAC algorithm with adaptive iteration count.

    Args:
        matches_kps1, matches_kps2 (np.array): Nx2 arrays of matched (x, y) coordinates.
        max_iter (int): Maximum number of RANSAC iterations.
        threshold (float): Maximum error (in pixels) for a point to be an inlier.
        min_inliers (int): Minimum number of inliers required for a valid model.

    Returns:
        tuple: (best_H, inliers_indices)
    """
    num_matches = matches_kps1.shape[0]
    
    if num_matches < 4:
        return np.identity(3), np.array([])
    
    best_H = np.identity(3)
    max_inliers = 0
    best_inliers_indices = np.array([])
    
    # Adaptive RANSAC parameters
    confidence = 0.99  # Confidence level
    iter_count = 0
    
    while iter_count < max_iter:
        # 1. Sample: Randomly select 4 pairs
        random_indices = np.random.choice(num_matches, 4, replace=False)
        p1_sample = matches_kps1[random_indices]
        p2_sample = matches_kps2[random_indices]

        # 2. Hypothesize: Compute Homography H
        try:
            H_model = compute_homography(p1_sample, p2_sample)
        except np.linalg.LinAlgError:
            iter_count += 1
            continue

        # 3. Test: Transform all points in p1 using H and measure error

        # Convert to homogeneous coordinates
        p1_hom = np.hstack((matches_kps1, np.ones((num_matches, 1))))

        # Transform p1' = H * p1^T (matrix multiplication)
        p2_prime_hom = (H_model @ p1_hom.T).T

        # Convert back to Cartesian (p2_prime_x = p2_prime_hom_x / p2_prime_hom_w)
        # Avoid division by zero
        w = p2_prime_hom[:, 2]
        valid_mask = np.abs(w) > 1e-6
        p2_prime_cartesian = np.zeros((num_matches, 2))
        p2_prime_cartesian[valid_mask] = p2_prime_hom[valid_mask, :2] / w[valid_mask, np.newaxis]

        # Calculate Euclidean distance error (residual)
        errors = np.sqrt(np.sum((matches_kps2 - p2_prime_cartesian) ** 2, axis=1))
        errors[~valid_mask] = np.inf  # Mark invalid points as outliers

        # Find Inliers
        inliers_indices = np.where(errors < threshold)[0]
        current_inliers = len(inliers_indices)

        # 4. Consensus: Check if this model is better
        if current_inliers > max_inliers and current_inliers >= min_inliers:
            max_inliers = current_inliers

            # Refine the Homography using all inliers
            if current_inliers >= 4:
                best_inlier_p1 = matches_kps1[inliers_indices]
                best_inlier_p2 = matches_kps2[inliers_indices]
                try:
                    best_H = compute_homography(best_inlier_p1, best_inlier_p2)
                    best_inliers_indices = inliers_indices
                except np.linalg.LinAlgError:
                    pass

        # Adaptive iteration count: Stop early if we have enough confidence
        if max_inliers > 0:
            inlier_ratio = max_inliers / num_matches
            if inlier_ratio > 0.5:  # If more than 50% are inliers, we're confident
                # Calculate required iterations: N = log(1-p) / log(1-(1-epsilon)^s)
                # where p=confidence, epsilon=outlier ratio, s=sample size (4)
                epsilon = 1 - inlier_ratio
                if epsilon > 0:
                    required_iter = np.log(1 - confidence) / np.log(1 - (1 - epsilon) ** 4)
                    if iter_count > required_iter and max_inliers >= min_inliers:
                        break

        iter_count += 1

    # Final verification: find inliers for the best H
    if max_inliers > 0:
        final_inliers_indices = find_inliers_for_H(best_H, matches_kps1, matches_kps2, threshold)
        return best_H, final_inliers_indices
    
    return best_H, best_inliers_indices


def find_inliers_for_H(H, p1, p2, threshold):
    """Helper function to find the indices of inliers for a given H."""
    num_matches = p1.shape[0]
    p1_hom = np.hstack((p1, np.ones((num_matches, 1))))
    p2_prime_hom = (H @ p1_hom.T).T
    
    # Avoid division by zero
    w = p2_prime_hom[:, 2]
    valid_mask = np.abs(w) > 1e-6
    p2_prime_cartesian = np.zeros((num_matches, 2))
    p2_prime_cartesian[valid_mask] = p2_prime_hom[valid_mask, :2] / w[valid_mask, np.newaxis]
    
    errors = np.sqrt(np.sum((p2 - p2_prime_cartesian) ** 2, axis=1))
    errors[~valid_mask] = np.inf
    
    return np.where(errors < threshold)[0]


if __name__ == '__main__':
    # Example usage for testing
    pass
