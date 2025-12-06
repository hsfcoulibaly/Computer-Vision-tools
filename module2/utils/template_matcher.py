import cv2
import numpy as np
import os


def find_match_and_blur(image_path, template_path):
    """
    Finds a specific template in the image using multi-scale matching and feature detection.
    
    Args:
        image_path: Path to the scene image
        template_path: Path to the specific template to match
        
    Returns:
        tuple: (processed_image, message, match_found)
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, "Error: Could not load source image.", False

    if not os.path.exists(template_path):
        return None, f"Error: Template not found at {template_path}", False

    template_bgr = cv2.imread(template_path)
    if template_bgr is None:
        return None, "Error: Could not load template image.", False
    
    # Try multi-scale template matching first (with lower threshold)
    result, message, match_found = multi_scale_template_match(img_bgr.copy(), template_bgr, threshold=0.4)
    
    if match_found:
        return result, message, True
    
    # If template matching fails, try feature-based matching (ORB)
    result, message, match_found = feature_based_match(img_bgr.copy(), template_bgr, min_matches=6)
    
    if match_found:
        return result, message, True
    
    # Return the original image with failure message
    return img_bgr, message, False


def multi_scale_template_match(img_bgr, template_bgr, threshold=0.4):
    """
    Performs template matching at multiple scales to handle size differences.
    
    Args:
        img_bgr: Source image (BGR)
        template_bgr: Template image (BGR)
        threshold: Detection threshold (default 0.4, lowered for flexibility)
        
    Returns:
        tuple: (processed_image, message, match_found)
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    template_h, template_w = template_gray.shape[:2]
    img_h, img_w = img_gray.shape[:2]
    
    best_match = {
        'max_val': -1,
        'top_left': None,
        'scale': 1.0,
        'template_w': template_w,
        'template_h': template_h
    }
    
    # Try multiple scales (from 20% to 150% of original template size)
    # More scales = more accurate but slower
    scales = np.linspace(0.2, 1.5, 30)
    
    for scale in scales:
        # Resize template
        new_w = int(template_w * scale)
        new_h = int(template_h * scale)
        
        # Skip if template becomes too small or larger than source image
        if new_w < 20 or new_h < 20:
            continue
        if new_w > img_w or new_h > img_h:
            continue
            
        resized_template = cv2.resize(template_gray, (new_w, new_h))
        
        # Perform template matching
        result = cv2.matchTemplate(img_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_match['max_val']:
            best_match.update({
                'max_val': max_val,
                'top_left': max_loc,
                'scale': scale,
                'template_w': new_w,
                'template_h': new_h
            })
    
    if best_match['max_val'] >= threshold and best_match['top_left'] is not None:
        top_left = best_match['top_left']
        w, h = best_match['template_w'], best_match['template_h']
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # Ensure ROI is within image bounds
        x1, y1 = max(0, top_left[0]), max(0, top_left[1])
        x2, y2 = min(img_bgr.shape[1], bottom_right[0]), min(img_bgr.shape[0], bottom_right[1])
        
        # Extract and blur ROI
        roi = img_bgr[y1:y2, x1:x2]
        if roi.size > 0:
            blurred_roi = cv2.GaussianBlur(roi, (41, 41), 0)
            img_bgr[y1:y2, x1:x2] = blurred_roi
        
        # Draw bounding box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        return img_bgr, f"Object detected! Score: {best_match['max_val']:.2f}, Scale: {best_match['scale']:.2f}x", True
    
    return img_bgr, f"Template matching: No match found. Best score: {best_match['max_val']:.2f} (threshold: {threshold})", False


def feature_based_match(img_bgr, template_bgr, min_matches=6):
    """
    Uses ORB feature detection and matching for more robust object detection.
    Works better with rotation and scale changes.
    
    Args:
        img_bgr: Source image (BGR)
        template_bgr: Template image (BGR)
        min_matches: Minimum number of good matches required (lowered to 6)
        
    Returns:
        tuple: (processed_image, message, match_found)
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
    
    # Create ORB detector with more features for better matching
    orb = cv2.ORB_create(nfeatures=3000)
    
    # Detect keypoints and compute descriptors
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(img_gray, None)
    
    if des1 is None or des2 is None:
        return img_bgr, "Feature detection: Could not extract features from images.", False
    
    if len(kp1) < 4 or len(kp2) < 4:
        return img_bgr, "Feature detection: Not enough keypoints found.", False
    
    # Use BFMatcher with Hamming distance (for ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    # Find matches using KNN
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except cv2.error:
        return img_bgr, "Feature matching: Could not compute matches.", False
    
    # Apply ratio test (Lowe's ratio test) - relaxed threshold
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.8 * n.distance:  # Relaxed from 0.75 to 0.8
                good_matches.append(m)
    
    if len(good_matches) < min_matches:
        return img_bgr, f"Feature matching: Only {len(good_matches)} matches found (need {min_matches}+).", False
    
    # Extract matching point coordinates
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    try:
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    except cv2.error:
        return img_bgr, "Feature matching: Could not compute homography.", False
    
    if M is None:
        return img_bgr, "Feature matching: Homography computation failed.", False
    
    # Count inliers - need at least 4 for a valid homography
    inliers = np.sum(mask) if mask is not None else len(good_matches)
    if inliers < 4:
        return img_bgr, f"Feature matching: Not enough inliers ({inliers}).", False
    
    # Get template dimensions and transform corners
    h, w = template_gray.shape
    pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    
    try:
        dst = cv2.perspectiveTransform(pts, M)
    except cv2.error:
        return img_bgr, "Feature matching: Perspective transform failed.", False
    
    # Validate the detected region (check if it's a valid quadrilateral)
    dst_reshaped = dst.reshape(4, 2)
    
    # Check for valid polygon (area should be reasonable)
    area = cv2.contourArea(dst_reshaped.astype(np.int32))
    img_area = img_bgr.shape[0] * img_bgr.shape[1]
    min_area = 500  # Lowered minimum area
    max_area = img_area * 0.5  # Max 50% of image (was 90%)
    
    if area < min_area:
        return img_bgr, f"Feature matching: Detected region too small (area: {area:.0f}).", False
    
    if area > max_area:
        # If area is too large, the match is probably wrong - skip feature matching
        return img_bgr, f"Feature matching: Detected region too large (area: {area:.0f}, max: {max_area:.0f}).", False
    
    # Additional validation: check if the polygon is convex and has reasonable aspect ratio
    if not cv2.isContourConvex(dst_reshaped.astype(np.int32)):
        # Try to use bounding box instead for non-convex results
        pass  # Continue anyway, we'll use the bounding box
    
    # Get bounding rectangle
    x, y, bw, bh = cv2.boundingRect(dst_reshaped.astype(np.int32))
    
    # Check aspect ratio - shouldn't be too extreme
    if bw > 0 and bh > 0:
        aspect_ratio = max(bw, bh) / min(bw, bh)
        if aspect_ratio > 10:  # Too elongated, probably a bad match
            return img_bgr, f"Feature matching: Invalid aspect ratio ({aspect_ratio:.1f}).", False
    
    # Ensure bounds are within image
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(img_bgr.shape[1], x + bw), min(img_bgr.shape[0], y + bh)
    
    # Make sure we have a valid region
    if x2 <= x1 or y2 <= y1:
        return img_bgr, "Feature matching: Invalid region bounds.", False
    
    # Extract and blur ROI
    roi = img_bgr[y1:y2, x1:x2]
    if roi.size > 0:
        blurred_roi = cv2.GaussianBlur(roi, (41, 41), 0)
        img_bgr[y1:y2, x1:x2] = blurred_roi
    
    # Draw the detected polygon outline
    img_bgr = cv2.polylines(img_bgr, [np.int32(dst)], True, (0, 255, 0), 3)
    
    return img_bgr, f"Object detected via feature matching! {len(good_matches)} matches, {inliers} inliers.", True


def find_best_match_and_blur(image_path, templates_dir):
    """
    Legacy function: Finds the best matching template from all templates in directory.
    Now uses multi-scale and feature-based matching.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None, "Error: Could not load source image."

    best_result = {
        'score': -1,
        'image': img_bgr.copy(),
        'template_name': '',
        'message': 'No templates found.'
    }

    # Iterate over all templates (obj1 to obj10)
    for i in range(1, 11):
        for ext in ['.jpg', '.jpeg', '.png']:
            template_filename = f'obj{i}{ext}'
            template_path = os.path.join(templates_dir, template_filename)

            if not os.path.exists(template_path):
                continue

            template_bgr = cv2.imread(template_path)
            if template_bgr is None:
                continue
            
            # Try multi-scale template matching
            result_img, message, match_found = multi_scale_template_match(
                img_bgr.copy(), template_bgr, threshold=0.4
            )
            
            if match_found:
                # Extract score from message
                import re
                score_match = re.search(r'Score: ([\d.]+)', message)
                score = float(score_match.group(1)) if score_match else 0.5
                
                if score > best_result['score']:
                    best_result.update({
                        'score': score,
                        'image': result_img,
                        'template_name': template_filename,
                        'message': message
                    })
            
            break  # Found template with this number, move to next

    if best_result['score'] > 0:
        return best_result['image'], f"Detected {best_result['template_name']}: {best_result['message']}"

    # If no template match, try feature-based on all templates
    for i in range(1, 11):
        for ext in ['.jpg', '.jpeg', '.png']:
            template_filename = f'obj{i}{ext}'
            template_path = os.path.join(templates_dir, template_filename)

            if not os.path.exists(template_path):
                continue

            template_bgr = cv2.imread(template_path)
            if template_bgr is None:
                continue
            
            result_img, message, match_found = feature_based_match(
                img_bgr.copy(), template_bgr, min_matches=6
            )
            
            if match_found:
                return result_img, f"Detected {template_filename} (feature matching): {message}"
            
            break

    return img_bgr, "No object detected with any template."
