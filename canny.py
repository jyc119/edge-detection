import numpy as np
import cv2 as cv
from scipy.ndimage import convolve


def gaussian_filter(img, size, sigma):
    kernel_1D = np.linspace(-(size // 2), size // 2, size)
    for i in range(size):
        kernel_1D[i] = np.exp(-(kernel_1D[i] ** 2) / (2 * sigma ** 2))
    kernel_1D /= kernel_1D.sum()

    kernel_2D = np.outer(kernel_1D, kernel_1D)
    kernel_2D /= kernel_2D.sum()
    print(kernel_2D)
    img_gaussian = cv.GaussianBlur(img, (5, 5), 1)
    cv.imshow("Gaussian kernel", img_gaussian)
    cv.waitKey(0)
    cv.imshow("Gaussian kernel", convolve(img, kernel_2D))
    cv.waitKey(0)



def canny_edge_detector(img):
    gaussian_filter(img, 5, 1)

    # Find intensity gradients

    # Apply magnitude thresholding

    # Apply double threshold

    # Track edge by hysteresis
    final_img = img
    return final_img


def main():
    print("Canny Edge Detection Test")

    # Convert to grayscale
    img = cv.imread('images/test_image_1.png', 0)
    img = canny_edge_detector(img)

    # Compare to OpenCV Canny
    # img = cv.Canny(img, 100, 200)
    # cv.imshow('Canny Edge Detector', img)
    # cv.waitKey(0)


if __name__ == '__main__':
    main()
