import numpy as np
import cv2
from scipy.ndimage import convolve
import matplotlib.pyplot as plt


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
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    Gy = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    Ix = convolve(img, Gx)
    Iy = convolve(img, Gy)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def magnitude_thresholding(G, theta):
    height, width = G.shape
    Z = np.zeros((height, width), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for x in range(1, height-1):
        for y in range(1, width-1):
            try:
                q = 255
                r = 255

                if (0 <= angle[x, y] < 22.5) or (157.5 <= angle[x, y] <= 180):
                    q = G[x, y+1]
                    r = G[x, y-1]
                elif 22.5 <= angle[x, y] < 67.5:
                    q = G[x+1, y-1]
                    r = G[x-1, y+1]
                elif 67.5 <= angle[x, y] < 112.5:
                    q = G[x+1, y]
                    r = G[x-1, y]
                elif 112.5 <= angle[x, y] < 157.5:
                    q = G[x-1, y-1]
                    r = G[x+1, y+1]
                if (G[x, y] >= q) and (G[x, y] >= r):
                    Z[x, y] = G[x, y]
                else:
                    Z[x, y] = 0
            except IndexError:
                pass

    return Z


def double_threshold(img, low, high):
    high = img.max() * high
    low = high * low

    height, width = img.shape
    res = np.zeros((height, width), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_x, strong_y = np.where(img >= high)
    # zeros_x, zeros_y = np.where(img < low)
    weak_x, weak_y = np.where((img <= high) & (img >= low))

    res[strong_x, strong_y] = strong
    # res[zeros_x, zeros_y] = 0
    res[weak_x, weak_y] = weak

    return res, weak, strong


def hysteresis(img, weak, strong=200):
    height, width = img.shape
    for x in range(1, height-1):
        for y in range(1, width-1):
            if img[x, y] == weak:
                try:
                    if ((img[x+1, y-1] == strong) or (img[x+1, y] == strong) or
                        (img[x+1, y+1] == strong) or (img[x, y-1] == strong) or
                        (img[x, y+1] == strong) or (img[x-1, y-1] == strong) or
                        (img[x-1, y] == strong) or (img[x-1, y+1] == strong)):
                        img[x, y] = strong
                    else:
                        img[x, y] = 0
                except IndexError:
                    pass
    return img


def canny_edge_detector(img):
    # Apply gaussian filter (equivalent to gaussianBlur)
    gaussian_kernel = gaussian_filter_kernel(5, 1)
    gaussian_img = convolve(img, gaussian_kernel)

    # Find intensity gradients
    G, theta = gradient_intesity(gaussian_img)

    # Apply magnitude thresholding
    Z = magnitude_thresholding(G, theta)

    # Apply double threshold
    threshold_img, weak, strong = double_threshold(Z, 0.3, 0.45)

    # Track edge by hysteresis
    final_img = hysteresis(threshold_img, weak)
    return final_img


def main():
    print("Canny Edge Detection Test")

    # Convert to grayscale
    img = cv2.imread('images/test_image_1.png', 0)

    edges = canny_edge_detector(img)
    edges = np.uint8(edges)
    cv2.imshow('Canny from Scratch', edges)
    cv2.waitKey(0)

    # Compare to OpenCV Canny
    img = cv2.Canny(img, 100, 200)
    cv2.imshow('Canny Edge Detector', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
