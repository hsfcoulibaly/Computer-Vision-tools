import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

# Import custom algorithms
# NOTE: Assume these are correctly imported based on your project structure
from .sift import compute_sift_features, match_features
from .ransac import ransac_homography, compute_homography


def custom_stitch_images(image_paths):
    """
    Main function to stitch a list of images into a panorama.
    """
    # Store the images as NumPy arrays for the WARPING/BLENDING step later
    images_data = [np.array(Image.open(p).convert('RGB')) for p in image_paths]

    # We choose the middle image as the reference frame.
    center_idx = len(image_paths) // 2
    H_total = {center_idx: np.identity(3)}

    # --- Chain Stitching: Left side (i -> i+1) ---
    for i in range(center_idx, len(image_paths) - 1):
        print(f"Stitching image {i} to {i + 1}...")

        # Pass the paths to find_homography for feature extraction
        H_i_to_iplus1 = find_homography(image_paths[i], image_paths[i + 1])

        # H_i+1_to_center = H_i+1_to_i * H_i_to_center
        # H_i+1_to_i = inv(H_i_to_iplus1)
        H_total[i + 1] = np.linalg.inv(H_i_to_iplus1) @ H_total[i]

    # --- Chain Stitching: Right side (i -> i-1) ---
    for i in range(center_idx, 0, -1):
        print(f"Stitching image {i} to {i - 1}...")

        # Pass the paths to find_homography for feature extraction
        H_i_to_iminus1 = find_homography(image_paths[i], image_paths[i - 1])

        # H_i-1_to_center = H_i-1_to_i * H_i_to_center
        # H_i-1_to_i = inv(H_i_to_iminus1)
        H_total[i - 1] = np.linalg.inv(H_i_to_iminus1) @ H_total[i]

    # 2. Compute Canvas Size
    min_x, max_x, min_y, max_y = calculate_canvas_bounds(images_data, H_total)

    # Calculate the shift matrix T to bring min_x, min_y to (0, 0)
    width = int(round(max_x - min_x))
    height = int(round(max_y - min_y))
    output_shape = (height, width, 3)

    T_shift = np.array([
        [1, 0, -min_x],
        [0, 1, -min_y],
        [0, 0, 1]
    ])

    # 3. Warping and Blending (Use images_data)
    stitched_image = blend_images(images_data, H_total, T_shift, output_shape)

    # 4. Save result
    result_path = 'static/results/stitched_custom.jpg'
    Image.fromarray(stitched_image.astype(np.uint8)).save(result_path)
    return result_path


def find_homography(img1_path, img2_path):
    """SIFT, Match, and RANSAC for a single image pair."""
    # 1. SIFT Feature Extraction
    kps1_np, descs1 = compute_sift_features(img1_path)
    kps2_np, descs2 = compute_sift_features(img2_path)

    # 2. Feature Matching
    raw_matches = match_features(descs1, descs2)

    # --- RANSAC Check ---
    if len(raw_matches) < 4:
        # Fallback: If not enough matches, return identity matrix (will likely fail stitching)
        print("Warning: Less than 4 matches found. Returning Identity Homography.")
        return np.identity(3)

    # Extract matched coordinates
    # kps1_np and kps2_np are already (x, y) coordinates from compute_sift_features
    src_pts = np.float32([kps1_np[m[0]] for m in raw_matches]).reshape(-1, 2)
    dst_pts = np.float32([kps2_np[m[1]] for m in raw_matches]).reshape(-1, 2)

    # 3. RANSAC Optimization
    H, inliers = ransac_homography(src_pts, dst_pts)

    return H


def calculate_canvas_bounds(images, H_total):
    """Determines the size of the final panorama canvas."""
    all_corners = []

    for i, img in enumerate(images):
        H = H_total[i]
        h, w = img.shape[:2]
        corners = np.array([
            [0, 0, 1], [w - 1, 0, 1],
            [0, h - 1, 1], [w - 1, h - 1, 1]
        ]).T  # 3x4 matrix

        # Transform corners p' = H * p
        transformed_corners_hom = H @ corners

        # Convert to Cartesian
        transformed_corners_cart = transformed_corners_hom[:2] / transformed_corners_hom[2]
        all_corners.append(transformed_corners_cart)

    all_points = np.hstack(all_corners)

    min_x, max_x = np.min(all_points[0]), np.max(all_points[0])
    min_y, max_y = np.min(all_points[1]), np.max(all_points[1])

    return min_x, max_x, min_y, max_y


def blend_images(images, H_total, T_shift, output_shape):
    """Applies the inverse warp and blends the images."""
    final_canvas = np.zeros(output_shape, dtype=np.float32)
    weight_map = np.zeros(output_shape, dtype=np.float32)

    # Generate output grid coordinates (r, c)
    r_coords, c_coords = np.indices(output_shape[:2])
    output_coords_hom = np.stack([c_coords.flatten(), r_coords.flatten(), np.ones(c_coords.size)])

    for i, img in enumerate(images):
        # Full transformation from final canvas to source image: H_inv * T_inv
        # Since T_shift shifts the final panorama, the transformation to the source is (T_shift @ H_total[i])^-1
        H_combined = T_shift @ H_total[i]
        try:
            H_total_inv = np.linalg.inv(H_combined)
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix for image {i}. Skipping warp.")
            continue

        # Transform output coordinates back to the source image (p = H_inv * p')
        source_coords_hom = H_total_inv @ output_coords_hom
        source_coords_cart = source_coords_hom[:2] / source_coords_hom[2]

        # Use inverse warping (map_coordinates) to sample from the source image
        warped_img = np.zeros(output_shape, dtype=np.float32)
        h, w = img.shape[:2]

        # Determine valid points (within source image boundaries)
        x_src = source_coords_cart[0].reshape(output_shape[:2])
        y_src = source_coords_cart[1].reshape(output_shape[:2])

        valid_mask = (x_src >= 0) & (x_src < w - 1) & (y_src >= 0) & (y_src < h - 1)

        # Warp each channel using map_coordinates
        for channel in range(3):
            # map_coordinates expects (y, x) order, so pass [y_src, x_src]
            warped_pixels = map_coordinates(
                img[:, :, channel],
                [y_src, x_src],
                order=1,  # Linear interpolation
                mode='constant',
                cval=0.0
            )

            # Apply valid mask to the interpolated pixels
            warped_img[:, :, channel] = warped_pixels.reshape(output_shape[:2]) * valid_mask

            # Simple Accumulator Blending
        current_mask = valid_mask[:, :, np.newaxis]

        # Accumulate image and weight
        final_canvas += warped_img
        weight_map += current_mask

    # Normalize the canvas by the weight map (avoid division by zero)
    final_canvas /= np.maximum(weight_map, 1e-6)

    return final_canvas


if __name__ == '__main__':
    # Example usage for testing
    pass
