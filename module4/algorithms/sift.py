import numpy as np
from scipy import ndimage
from PIL import Image

# Global parameters (You can tune these)
NUM_OCTAVES = 4
SCALES_PER_OCTAVE = 5
INITIAL_SIGMA = 1.6
K = 2 ** (1.0 / SCALES_PER_OCTAVE)  # Scale factor between blurs

# Keypoint refinement thresholds
CONTRAST_THRESHOLD = 0.03  # Low contrast threshold
EDGE_RATIO_THRESHOLD = 10.0  # Edge response threshold (r = (Tr(H))^2 / Det(H))


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

    # 2. Keypoint Localization with refinement
    keypoints = find_keypoints(dog_pyramid, gaussian_pyramid)

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


# --- SIFT Step 2: Keypoint Localization with Refinement ---

def refine_keypoint(dog_pyramid, octave_idx, scale_idx, r, c):
    """
    Refines keypoint location using Taylor expansion (sub-pixel/sub-scale accuracy).
    Returns refined (x, y, sigma) or None if keypoint should be rejected.
    """
    img = dog_pyramid[octave_idx][scale_idx]
    h, w = img.shape
    
    # Check bounds
    if r < 1 or r >= h - 1 or c < 1 or c >= w - 1:
        return None
    
    # Compute derivatives using finite differences
    D = img[r, c]
    Dx = (img[r, c + 1] - img[r, c - 1]) * 0.5
    Dy = (img[r + 1, c] - img[r - 1, c]) * 0.5
    
    # Get scale derivatives if available
    if scale_idx > 0 and scale_idx < len(dog_pyramid[octave_idx]) - 1:
        img_prev = dog_pyramid[octave_idx][scale_idx - 1]
        img_next = dog_pyramid[octave_idx][scale_idx + 1]
        Ds = (img_next[r, c] - img_prev[r, c]) * 0.5
    else:
        Ds = 0.0
    
    # Second derivatives
    Dxx = img[r, c + 1] - 2 * D + img[r, c - 1]
    Dyy = img[r + 1, c] - 2 * D + img[r - 1, c]
    Dxy = ((img[r + 1, c + 1] - img[r + 1, c - 1]) - (img[r - 1, c + 1] - img[r - 1, c - 1])) * 0.25
    
    # Hessian matrix
    H = np.array([[Dxx, Dxy], [Dxy, Dyy]])
    
    # Solve for offset: H * offset = -gradient
    try:
        offset = -np.linalg.solve(H, np.array([Dx, Dy]))
        x_offset, y_offset = offset[0], offset[1]
    except np.linalg.LinAlgError:
        return None
    
    # Reject if offset is too large (> 0.5 pixels)
    if abs(x_offset) > 0.5 or abs(y_offset) > 0.5:
        return None
    
    # Interpolated value at refined location
    D_refined = D + 0.5 * (Dx * x_offset + Dy * y_offset)
    
    # Reject low contrast points
    if abs(D_refined) < CONTRAST_THRESHOLD:
        return None
    
    # Edge rejection using Hessian
    trace_H = Dxx + Dyy
    det_H = Dxx * Dyy - Dxy * Dxy
    
    if det_H <= 0:
        return None
    
    edge_response = (trace_H ** 2) / det_H
    if edge_response > ((EDGE_RATIO_THRESHOLD + 1) ** 2) / EDGE_RATIO_THRESHOLD:
        return None
    
    # Return refined coordinates
    refined_x = c + x_offset
    refined_y = r + y_offset
    refined_scale = scale_idx  # Could also refine scale, but keeping it simple
    
    # Store both octave coordinates (for descriptor computation) and original image coordinates
    return {
        'octave': octave_idx,
        'scale': refined_scale,
        'row': refined_y,  # In octave coordinates
        'col': refined_x,  # In octave coordinates
        'x': refined_x * (2 ** octave_idx),  # In original image coordinates
        'y': refined_y * (2 ** octave_idx),  # In original image coordinates
        'sigma': INITIAL_SIGMA * (K ** refined_scale) * (2 ** octave_idx)
    }


def find_keypoints(dog_pyramid, gaussian_pyramid):
    """Detects local extrema in scale-space and refines their position."""
    keypoints = []
    for i, octave in enumerate(dog_pyramid):
        for s in range(1, len(octave) - 1):  # Compare current scale with neighbors
            img_c = octave[s]
            img_n = octave[s + 1] if s + 1 < len(octave) else None
            img_p = octave[s - 1] if s > 0 else None

            # Check if image is too small (e.g., from downsampling)
            if img_c.shape[0] < 3 or img_c.shape[1] < 3:
                continue

            for r in range(1, img_c.shape[0] - 1):
                for c in range(1, img_c.shape[1] - 1):
                    pixel_value = img_c[r, c]

                    # Check if pixel is a local extremum against all 26 neighbors
                    # (8 in current scale, 9 in scale above, 9 in scale below)
                    is_max_in_scale = pixel_value > np.max(img_c[r - 1:r + 2, c - 1:c + 2])
                    is_min_in_scale = pixel_value < np.min(img_c[r - 1:r + 2, c - 1:c + 2])
                    
                    is_extremum = False
                    if img_n is not None and img_p is not None:
                        is_max_above = pixel_value > np.max(img_n[r - 1:r + 2, c - 1:c + 2])
                        is_max_below = pixel_value > np.max(img_p[r - 1:r + 2, c - 1:c + 2])
                        is_min_above = pixel_value < np.min(img_n[r - 1:r + 2, c - 1:c + 2])
                        is_min_below = pixel_value < np.min(img_p[r - 1:r + 2, c - 1:c + 2])
                        
                        is_extremum = (pixel_value > 0 and is_max_in_scale and is_max_above and is_max_below) or \
                                     (pixel_value < 0 and is_min_in_scale and is_min_above and is_min_below)
                    else:
                        is_extremum = is_max_in_scale or is_min_in_scale

                    if is_extremum:
                        # Refine keypoint location
                        refined_kp = refine_keypoint(dog_pyramid, i, s, r, c)
                        if refined_kp is not None:
                            keypoints.append(refined_kp)

    return keypoints


# --- SIFT Step 3 & 4: Orientation and Descriptor ---

def compute_gradients(img):
    """Compute gradient magnitude and orientation for an image."""
    # Sobel operators for gradients
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    
    grad_x = ndimage.convolve(img, sobel_x)
    grad_y = ndimage.convolve(img, sobel_y)
    
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    orientation = np.arctan2(grad_y, grad_x)  # Returns in range [-pi, pi]
    
    return magnitude, orientation


def assign_orientation(keypoint, gaussian_pyramid):
    """Assign dominant orientation(s) to a keypoint using 36-bin histogram."""
    octave_idx = keypoint['octave']
    scale_idx = keypoint['scale']
    
    # Use the Gaussian image at the keypoint's scale
    # SIFT uses the scale below DoG scale, so scale_idx + 1
    if scale_idx + 1 >= len(gaussian_pyramid[octave_idx]):
        scale_idx = len(gaussian_pyramid[octave_idx]) - 2
    
    img = gaussian_pyramid[octave_idx][scale_idx + 1]
    
    # Get keypoint location in this octave
    r = int(round(keypoint['row']))
    c = int(round(keypoint['col']))
    
    # Compute gradients
    magnitude, orientation = compute_gradients(img)
    
    # Create 36-bin histogram (10 degrees per bin)
    hist = np.zeros(36)
    
    # Window size: 1.5 * sigma (typically 16 pixels radius)
    sigma = keypoint.get('sigma', INITIAL_SIGMA * (K ** scale_idx))
    radius = int(round(3 * sigma))
    
    h, w = img.shape
    
    # Build orientation histogram with Gaussian weighting
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            y, x = r + dy, c + dx
            
            if y < 0 or y >= h or x < 0 or x >= w:
                continue
            
            # Gaussian weight (distance from keypoint)
            dist_sq = dx * dx + dy * dy
            weight = np.exp(-dist_sq / (2 * (1.5 * sigma) ** 2))
            
            # Add to histogram
            angle = orientation[y, x]
            if angle < 0:
                angle += 2 * np.pi  # Convert to [0, 2*pi]
            
            bin_idx = int(angle / (2 * np.pi / 36)) % 36
            hist[bin_idx] += magnitude[y, x] * weight
    
    # Find dominant orientation(s)
    max_val = np.max(hist)
    orientations = []
    
    for i in range(36):
        # Check if this bin is a peak
        prev_bin = hist[(i - 1) % 36]
        next_bin = hist[(i + 1) % 36]
        curr_bin = hist[i]
        
        if curr_bin > 0.8 * max_val and curr_bin > prev_bin and curr_bin > next_bin:
            # Interpolate peak location
            if prev_bin > next_bin:
                offset = 0.5 * (prev_bin - next_bin) / (prev_bin - curr_bin + 1e-6)
            else:
                offset = 0.5 * (next_bin - prev_bin) / (next_bin - curr_bin + 1e-6)
            
            angle = (i + offset) * (2 * np.pi / 36)
            orientations.append(angle)
    
    # If no peak found, use maximum
    if len(orientations) == 0:
        max_idx = np.argmax(hist)
        orientations.append(max_idx * (2 * np.pi / 36))
    
    return orientations


def compute_descriptor(keypoint, orientation, gaussian_pyramid):
    """Compute 128-element SIFT descriptor for a keypoint."""
    octave_idx = keypoint['octave']
    scale_idx = keypoint['scale']
    
    # Use the Gaussian image at the keypoint's scale
    if scale_idx + 1 >= len(gaussian_pyramid[octave_idx]):
        scale_idx = len(gaussian_pyramid[octave_idx]) - 2
    
    img = gaussian_pyramid[octave_idx][scale_idx + 1]
    
    # Get keypoint location
    r = int(round(keypoint['row']))
    c = int(round(keypoint['col']))
    
    # Compute gradients
    magnitude, grad_orientation = compute_gradients(img)
    
    # Rotate gradients relative to keypoint orientation
    cos_o = np.cos(orientation)
    sin_o = np.sin(orientation)
    
    # Descriptor parameters
    descriptor_size = 4  # 4x4 spatial bins
    num_bins = 8  # 8 orientation bins per spatial bin
    descriptor = np.zeros(descriptor_size * descriptor_size * num_bins)
    
    # Window size: typically 16x16 pixels
    half_width = 8
    h, w = img.shape
    
    for dy in range(-half_width, half_width):
        for dx in range(-half_width, half_width):
            # Rotate coordinates relative to keypoint orientation
            x_rot = dx * cos_o + dy * sin_o
            y_rot = -dx * sin_o + dy * cos_o
            x_orig = c + dx
            y_orig = r + dy
            
            if x_orig < 0 or x_orig >= w or y_orig < 0 or y_orig >= h:
                continue
            
            # Determine which 4x4 bin this pixel belongs to
            bin_x = int((x_rot + half_width) / (2 * half_width / descriptor_size))
            bin_y = int((y_rot + half_width) / (2 * half_width / descriptor_size))
            
            # Clamp to valid range
            bin_x = max(0, min(descriptor_size - 1, bin_x))
            bin_y = max(0, min(descriptor_size - 1, bin_y))
            
            # Get orientation relative to keypoint
            angle = grad_orientation[y_orig, x_orig] - orientation
            if angle < 0:
                angle += 2 * np.pi
            if angle >= 2 * np.pi:
                angle -= 2 * np.pi
            
            # Determine orientation bin
            orient_bin = int(angle / (2 * np.pi / num_bins)) % num_bins
            
            # Gaussian weight (distance from center)
            dist_sq = dx * dx + dy * dy
            weight = np.exp(-dist_sq / (2 * (half_width * 0.5) ** 2))
            
            # Add to descriptor
            desc_idx = (bin_y * descriptor_size + bin_x) * num_bins + orient_bin
            descriptor[desc_idx] += magnitude[y_orig, x_orig] * weight
    
    # Normalize descriptor
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor /= norm
    
    # Threshold and renormalize (SIFT robustness)
    descriptor = np.clip(descriptor, 0, 0.2)
    norm = np.linalg.norm(descriptor)
    if norm > 0:
        descriptor /= norm
    
    return descriptor


def generate_descriptors(keypoints, gaussian_pyramid):
    """Assigns orientation and computes the 128-element descriptor vector."""
    final_keypoints = []

    for kp in keypoints:
        # Assign orientation(s)
        orientations = assign_orientation(kp, gaussian_pyramid)
        
        # For each orientation, create a keypoint with descriptor
        for orient in orientations:
            # Compute descriptor
            descriptor = compute_descriptor(kp, orient, gaussian_pyramid)
            
            final_keypoints.append({
                'coords': (kp['x'], kp['y']),
                'descriptor': descriptor,
                'orientation': orient
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
