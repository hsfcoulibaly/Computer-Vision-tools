import numpy as np
from scipy import ndimage
from PIL import Image

# Global parameters (You can tune these)
NUM_OCTAVES = 4
SCALES_PER_OCTAVE = 5
INITIAL_SIGMA = 1.6
K = 2 ** (1.0 / SCALES_PER_OCTAVE)  # Scale factor between blurs


def compute_sift_features(image_path):
    """
    Main function to compute SIFT keypoints and descriptors from an image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        tuple: (keypoints, descriptors)
    """
    try:
        img_gray = load_and_preprocess_image(image_path)
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return np.array([]), np.array([])

    # 1. Scale-Space Construction
    gaussian_pyramid = build_gaussian_pyramid(img_gray)
    dog_pyramid = build_dog_pyramid(gaussian_pyramid)

    # 2. Keypoint Localization
    # NOTE: find_keypoints currently only performs local extrema detection;
    # Sub-pixel refinement, low-contrast, and edge rejection are still TODOs.
    keypoints = find_keypoints(dog_pyramid)

    # 3. Orientation Assignment & Descriptor Generation
    keypoints_with_desc = generate_descriptors(keypoints, gaussian_pyramid)

    # Separate keypoints and descriptors for output
    kps = np.array([kp['coords'] for kp in keypoints_with_desc])
    descs = np.array([kp['descriptor'] for kp in keypoints_with_desc])

    # If no keypoints are found, return empty arrays
    if len(kps) == 0:
        return np.array([]).reshape(0, 2), np.array([]).reshape(0, 128)

    return kps, descs


def load_and_preprocess_image(image_path):
    # Load image, convert to grayscale, and potentially double its size for the base level
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32) / 255.0
    # Optional: Initial upsampling for better scale-space coverage
    # img_upscaled = ndimage.zoom(img_array, 2, order=1)
    return img_array


# --- SIFT Step 1: Pyramid Construction ---

def build_gaussian_pyramid(img):
    """Creates a list of lists: Octave -> Scale -> Gaussian Image."""
    pyramid = []
    # Start with base image at initial sigma
    base_img = ndimage.gaussian_filter(img, sigma=INITIAL_SIGMA)

    for i in range(NUM_OCTAVES):
        octave = []
        sigma_prev = INITIAL_SIGMA  # Sigma for the current octave base

        for s in range(SCALES_PER_OCTAVE + 3):  # Need 3 extra scales for DoG calculation
            sigma_current = sigma_prev * K
            # Apply additional blur: sigma_eff^2 = sigma_current^2 - sigma_prev^2
            sigma_add = np.sqrt(sigma_current ** 2 - sigma_prev ** 2)

            if s == 0:
                octave.append(base_img)
            else:
                blurred_img = ndimage.gaussian_filter(octave[-1], sigma=sigma_add)
                octave.append(blurred_img)

            sigma_prev = sigma_current

        pyramid.append(octave)
        # Downsample the 3rd image (2 scales from base) to create the base for the next octave
        if i < NUM_OCTAVES - 1:
            base_img = octave[SCALES_PER_OCTAVE].copy()[::2, ::2]  # Subsample by 2

    return pyramid


def build_dog_pyramid(gaussian_pyramid):
    """Creates the Difference of Gaussians pyramid by subtracting adjacent images."""
    dog_pyramid = []
    for octave in gaussian_pyramid:
        dog_octave = [octave[i + 1] - octave[i] for i in range(len(octave) - 1)]
        dog_pyramid.append(dog_octave)
    return dog_pyramid


# --- SIFT Step 2: Keypoint Localization ---

def find_keypoints(dog_pyramid):
    """Detects local extrema in scale-space and refines their position."""
    keypoints = []
    for i, octave in enumerate(dog_pyramid):
        for s in range(1, len(octave) - 2):  # Compare current scale with neighbors
            img_c = octave[s]
            img_n = octave[s + 1]
            img_p = octave[s - 1]

            # Check if image is too small (e.g., from downsampling)
            if img_c.shape[0] < 3 or img_c.shape[1] < 3:
                continue

            for r in range(1, img_c.shape[0] - 1):
                for c in range(1, img_c.shape[1] - 1):
                    pixel_value = img_c[r, c]

                    # Check if pixel is a local extremum against all 26 neighbors
                    # (8 in current scale, 9 in scale above, 9 in scale below)
                    is_max_in_scale = pixel_value > np.max(img_c[r - 1:r + 2, c - 1:c + 2])
                    is_max_above = pixel_value > np.max(img_n[r - 1:r + 2, c - 1:c + 2])
                    is_max_below = pixel_value > np.max(img_p[r - 1:r + 2, c - 1:c + 2])

                    is_min_in_scale = pixel_value < np.min(img_c[r - 1:r + 2, c - 1:c + 2])
                    is_min_above = pixel_value < np.min(img_n[r - 1:r + 2, c - 1:c + 2])
                    is_min_below = pixel_value < np.min(img_p[r - 1:r + 2, c - 1:c + 2])

                    if (pixel_value > 0 and is_max_in_scale and is_max_above and is_max_below) or \
                            (pixel_value < 0 and is_min_in_scale and is_min_above and is_min_below):
                        # TODO: Refine keypoint location (subpixel/subscale accuracy)
                        # TODO: Discard low contrast points (threshold on D(x))
                        # TODO: Discard points on edges (Hessian matrix check)

                        # Store provisional keypoint: (octave, scale, row, col)
                        keypoints.append({'octave': i, 'scale': s, 'row': r, 'col': c})

    return keypoints


# --- SIFT Step 3 & 4: Orientation and Descriptor ---

def generate_descriptors(keypoints, gaussian_pyramid):
    """Assigns orientation and computes the 128-element descriptor vector."""
    final_keypoints = []

    for kp in keypoints:
        octave_index = kp['octave']
        # scale_index = kp['scale']
        r, c = kp['row'], kp['col']

        # Image for gradient calculation is the corresponding Gaussian image
        # NOTE: SIFT specifies using the scale *below* the DoG scale for orientation/descriptor.
        # Since we use DoG[s], the corresponding Gaussian image is Gaussian[s+1].
        # In this current implementation, we'll simplify and use the same index for now.
        # img = gaussian_pyramid[octave_index][scale_index + 1] # More accurate SIFT

        # Placeholder descriptor generation (must be implemented from scratch):

        # 1. Orientation Assignment: Find dominant orientation(s) using a 36-bin histogram.
        # 2. Descriptor Generation: Calculate a 128-element vector from a 16x16 region (4x4 blocks * 8 bins).

        # Placeholder for final descriptor structure (128-D vector)
        descriptor = np.random.rand(128)  # Replace with actual descriptor calculation
        descriptor /= np.linalg.norm(descriptor)  # Normalize (essential step)

        final_keypoints.append({
            'coords': (c * (2 ** octave_index), r * (2 ** octave_index)),
            # (x, y) coordinates relative to original image
            'descriptor': descriptor
        })

    return final_keypoints


# --- Feature Matching (used by Stitching) ---

def match_features(descriptors1, descriptors2, ratio_thresh=0.7):
    """
    Compares two sets of descriptors using Euclidean distance and the
    ratio test (k-d tree or brute force).

    Returns:
        list: List of best matches (indices: (kps1_idx, kps2_idx))
    """
    if descriptors1.shape[0] == 0 or descriptors2.shape[0] == 0:
        return []

    matches = []
    # Brute force matching:
    for i, desc1 in enumerate(descriptors1):
        # Calculate distances to all desc2
        # Use broadcasting for distance calculation: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a.b
        # Or, simple difference squared sum:
        diff = descriptors2 - desc1
        distances = np.sqrt(np.sum(diff ** 2, axis=1))

        # Find the two closest neighbors
        sorted_indices = np.argsort(distances)

        # Ensure we have at least two neighbors to perform the ratio test
        if len(sorted_indices) < 2:
            continue

        best_match_idx = sorted_indices[0]
        second_best_match_idx = sorted_indices[1]

        # Apply the Lowe's ratio test (distance of best match / distance of second best match < threshold)
        if distances[best_match_idx] / distances[second_best_match_idx] < ratio_thresh:
            matches.append((i, best_match_idx))

    return matches


if __name__ == '__main__':
    # Example usage for testing
    pass
