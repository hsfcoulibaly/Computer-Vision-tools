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


def ransac_homography(matches_kps1, matches_kps2, max_iter=1000, threshold=5.0):
    """
    Estimates the best Homography matrix using the RANSAC algorithm.

    Args:
        matches_kps1, matches_kps2 (np.array): Nx2 arrays of matched (x, y) coordinates.
        max_iter (int): Maximum number of RANSAC iterations.
        threshold (float): Maximum error (in pixels) for a point to be an inlier.

    Returns:
        tuple: (best_H, inliers_indices)
    """
    num_matches = matches_kps1.shape[0]
    best_H = np.identity(3)
    max_inliers = 0

    for i in range(max_iter):
        # 1. Sample: Randomly select 4 pairs
        random_indices = np.random.choice(num_matches, 4, replace=False)
        p1_sample = matches_kps1[random_indices]
        p2_sample = matches_kps2[random_indices]

        # 2. Hypothesize: Compute Homography H
        H_model = compute_homography(p1_sample, p2_sample)

        # 3. Test: Transform all points in p1 using H and measure error

        # Convert to homogeneous coordinates
        p1_hom = np.hstack((matches_kps1, np.ones((num_matches, 1))))

        # Transform p1' = H * p1^T (matrix multiplication)
        p2_prime_hom = (H_model @ p1_hom.T).T

        # Convert back to Cartesian (p2_prime_x = p2_prime_hom_x / p2_prime_hom_w)
        p2_prime_cartesian = p2_prime_hom[:, :2] / p2_prime_hom[:, 2, np.newaxis]

        # Calculate Euclidean distance error (residual)
        errors = np.sqrt(np.sum((matches_kps2 - p2_prime_cartesian) ** 2, axis=1))

        # Find Inliers
        inliers_indices = np.where(errors < threshold)[0]
        current_inliers = len(inliers_indices)

        # 4. Consensus: Check if this model is better
        if current_inliers > max_inliers:
            max_inliers = current_inliers

            # Refine the Homography using all inliers
            best_inlier_p1 = matches_kps1[inliers_indices]
            best_inlier_p2 = matches_kps2[inliers_indices]
            best_H = compute_homography(best_inlier_p1, best_inlier_p2)

    # Final list of inliers based on the best H
    final_inliers_indices = find_inliers_for_H(best_H, matches_kps1, matches_kps2, threshold)

    return best_H, final_inliers_indices


def find_inliers_for_H(H, p1, p2, threshold):
    """Helper function to find the indices of inliers for a given H."""
    num_matches = p1.shape[0]
    p1_hom = np.hstack((p1, np.ones((num_matches, 1))))
    p2_prime_hom = (H @ p1_hom.T).T
    p2_prime_cartesian = p2_prime_hom[:, :2] / p2_prime_hom[:, 2, np.newaxis]
    errors = np.sqrt(np.sum((p2 - p2_prime_cartesian) ** 2, axis=1))
    return np.where(errors < threshold)[0]


if __name__ == '__main__':
    # Example usage for testing
    pass
