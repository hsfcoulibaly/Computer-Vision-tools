import cv2
import numpy as np
import os


def deconvolution_wiener(image_path, output_dir, kernel_size=15, sigma=5, K=0.001):
    """
    Applies Gaussian blur and then attempts deconvolution using a Wiener Filter.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save the output images
        kernel_size: Size of the Gaussian kernel (default: 15)
        sigma: Standard deviation of the Gaussian (default: 5)
        K: Wiener filter regularization parameter (default: 0.001)
        
    Returns:
        dict: Contains paths to original, blurred, and restored images, plus status message
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {'error': 'Error loading image.'}

    # Convert to float32 and normalize [0, 1] for best FFT arithmetic
    img_float = np.float32(img) / 255.0
    rows, cols = img_float.shape

    # 1. Convolution (Blurring) to create L_b
    blurred_img = cv2.GaussianBlur(img_float, (kernel_size, kernel_size), sigma)

    # --- Deconvolution Setup (Wiener Filter using NumPy FFT) ---

    # Construct and Transform the Gaussian Kernel G
    G_kernel = cv2.getGaussianKernel(kernel_size, sigma)
    G_kernel = G_kernel @ G_kernel.T

    # Pad the kernel to the same size as the image
    G_padded = np.zeros((rows, cols), dtype=np.float32)
    h_k, w_k = G_kernel.shape
    r_start, c_start = (rows - h_k) // 2, (cols - w_k) // 2
    G_padded[r_start: r_start + h_k, c_start: c_start + w_k] = G_kernel

    # Shift the zero-frequency component for numpy's fft2
    G_padded = np.fft.ifftshift(G_padded)

    # 2. Compute DFTs
    F_G = np.fft.fft2(G_padded)  # DFT of the Kernel G
    F_Lb = np.fft.fft2(blurred_img)  # DFT of the Blurred Image L_b

    # 3. Wiener Filter (Regularized Deconvolution)
    F_G_conj = np.conjugate(F_G)
    F_G_abs_sq = np.abs(F_G) ** 2

    # The filter transfer function H_w
    H_w = F_G_conj / (F_G_abs_sq + K)

    # Apply the filter in the frequency domain
    F_L_hat = H_w * F_Lb

    # 4. Inverse DFT to get restored image (L_hat)
    restored_img_normalized = np.fft.ifft2(F_L_hat)
    # Take the magnitude and convert back to 8-bit [0, 255]
    restored_img = np.abs(restored_img_normalized) * 255.0

    # --- Final Saving and Conversion ---

    # Convert back to uint8 for saving/displaying
    original_img_8u = np.clip(img_float * 255.0, 0, 255).astype(np.uint8)
    blurred_img_8u = np.clip(blurred_img * 255.0, 0, 255).astype(np.uint8)
    restored_img_8u = np.clip(restored_img, 0, 255).astype(np.uint8)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 5. Save Outputs
    original_path = os.path.join(output_dir, 'ft_original.png')
    blurred_path = os.path.join(output_dir, 'ft_blurred.png')
    restored_path = os.path.join(output_dir, 'ft_restored.png')

    cv2.imwrite(original_path, original_img_8u)
    cv2.imwrite(blurred_path, blurred_img_8u)
    cv2.imwrite(restored_path, restored_img_8u)

    return {
        'original': 'ft_original.png',
        'blurred': 'ft_blurred.png',
        'restored': 'ft_restored.png',
        'status': f"Deconvolution successful! Used: Gaussian Blur (K={kernel_size}, σ={sigma}) and Wiener Filter (K={K})."
    }
