import cv2
import numpy as np


def create_gaussian_kernel_ft(shape, sigma):
    """
    Creates a 2D Gaussian kernel and computes its 2D DFT.
    """
    rows, cols = shape
    center_row, center_col = rows // 2, cols // 2

    # Create the coordinate grids
    x = np.arange(cols)
    y = np.arange(rows)
    X, Y = np.meshgrid(x, y)

    # The Gaussian function
    gaussian = np.exp(-(((X - center_col) ** 2 + (Y - center_row) ** 2) / (2 * sigma ** 2)))
    gaussian /= gaussian.sum()  # Normalize

    # Shift the kernel to have the origin at (0, 0) for convolution/FT
    gaussian_shifted = np.fft.ifftshift(gaussian)

    # Compute the 2D DFT
    g_ft = np.fft.fft2(gaussian_shifted)
    return g_ft


def deblur_with_fourier_inverse(L_b, sigma, kernel_size):
    """
    Applies Gaussian blur (L -> L_b) and then uses inverse filtering
    (Fourier Transform) to retrieve L from L_b.

    L_recovered = F^-1 ( F(L_b) / F(G) )
    """

    # --- 1. Apply Gaussian Blurring (Creating L_b) ---
    L = L_b.copy()  # L_b is the input image in this script, rename for clarity
    L_b = cv2.GaussianBlur(L, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    L_b_gray = cv2.cvtColor(L_b, cv2.COLOR_BGR2GRAY).astype(np.float32)

    # --- 2. Create Gaussian Kernel FT (F(G)) ---
    rows, cols = L_b_gray.shape
    g_ft = create_gaussian_kernel_ft((rows, cols), sigma)

    # --- 3. Compute FT of Blurred Image (F(L_b)) ---
    # np.fft works better for the complex division step than cv2.dft
    l_b_ft = np.fft.fft2(L_b_gray)

    # --- 4. Deconvolution (Inverse Filtering) ---
    # F(L) = F(L_b) / F(G). Add a small epsilon to prevent division by zero/near-zero values
    epsilon = 1e-6
    # g_ft_regularized = g_ft.copy()
    # g_ft_regularized[np.abs(g_ft) < epsilon] = epsilon # Simple regularization

    # Perform the division in the frequency domain
    l_ft = l_b_ft / (g_ft + epsilon)

    # --- 5. Inverse FT (L_recovered) ---
    # Shift the zero-frequency component back to the center (optional, but good practice)
    # l_ft_shifted = np.fft.ifftshift(l_ft)
    L_recovered = np.fft.ifft2(l_ft)
    L_recovered = np.real(L_recovered)  # Only the real part is the image intensity

    # Normalize and convert to 8-bit image
    L_recovered = np.clip(L_recovered, 0, 255)
    L_recovered = L_recovered.astype(np.uint8)

    # Re-colorize (optional, for visualization)
    L_recovered_color = cv2.cvtColor(L_recovered, cv2.COLOR_GRAY2BGR)

    return L, L_b, L_recovered_color, "Deconvolution successful using Fourier Inverse Filter."

# Example function for Task 2 demonstration (can be run standalone)
# if __name__ == '__main__':
#     img = cv2.imread('test_image.jpg')
#     original, blurred, recovered, status = deblur_with_fourier_inverse(img, sigma=5, kernel_size=9)
#     cv2.imshow("Original", original)
#     cv2.imshow("Blurred (L_b)", blurred)
#     cv2.imshow("Recovered (L)", recovered)
#     cv2.waitKey(0)