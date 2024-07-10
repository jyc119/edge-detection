import numpy as np
import cv2 as cv
from scipy.ndimage import convolve


def gaussian_filter_kernel(size, sigma):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = np.exp(-(kernel_1D[i] ** 2) / (2 * sigma ** 2))
    kernel_1D /= kernel_1D.sum()

    kernel_2D = np.outer(kernel_1D, kernel_1D)
    kernel_2D /= kernel_2D.sum()

    return kernel_2D


def gradient_intesity(img):
    Gx = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    Gy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    Ix = convolve(img, Gx)
    Iy = convolve(img, Gy)

    G = np.hypot(Ix, Iy)
    G = np.clip(G, 0, 255)
    theta = np.arctan2(Iy, Ix)

    return G, theta


def magnitude_thresholding(G, theta):
    W, H = G.shape



def canny_edge_detector(img):
    # Apply gaussian filter (equivalent to gaussianBlur)
    gaussian_kernel = gaussian_filter_kernel(5, 1)
    gaussian_img = convolve(img, gaussian_kernel)

    # Find intensity gradients
    G, theta = gradient_intesity(gaussian_img)

    # Apply magnitude thresholding

    # Apply double threshold

    # Track edge by hysteresis
    final_img = gaussian_img
    return final_img


def main():
    print("Canny Edge Detection Test")

    # Convert to grayscale
    img = cv.imread('images/test_image_1.png', 0)

    img = canny_edge_detector(img)
    cv.imshow('Canny Edge Detector', img)
    cv.waitKey(0)

    # Compare to OpenCV Canny
    # img = cv.Canny(img, 100, 200)
    # cv.imshow('Canny Edge Detector', img)
    # cv.waitKey(0)


if __name__ == '__main__':
    main()
