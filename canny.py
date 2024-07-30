import numpy as np
import cv2


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

    Ix = cv2.filter2D(img, -1, Gx)
    Iy = cv2.filter2D(img, -1, Gy)

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

    strong_x, strong_y = np.where(img >= high)
    weak_x, weak_y = np.where((img <= high) & (img >= low))

    res[strong_x, strong_y] = 255
    res[weak_x, weak_y] = 50

    return res


def hysteresis(img, strong=255):
    height, width = img.shape
    for x in range(1, height-1):
        for y in range(1, width-1):
            if img[x, y] == 50:
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
    gaussian_kernel = gaussian_filter_kernel(3, 1)
    gaussian_img = cv2.filter2D(img, -1, gaussian_kernel)
    # gaussian_img = cv2.GaussianBlur(img, (3, 3), 1)

    # Find intensity gradients
    G, theta = gradient_intesity(gaussian_img)

    # Apply magnitude thresholding
    Z = magnitude_thresholding(G, theta)

    # Apply double threshold
    threshold_img = double_threshold(Z, 0.05, 0.1)

    # Track edge by hysteresis
    final_img = hysteresis(threshold_img)
    return final_img


def main():
    print("Canny Edge Detection Test")

    # Convert to grayscale
    img = cv2.imread('images/tomatoes.jpg', cv2.IMREAD_GRAYSCALE)

    cv2.imshow('Original Image- Grayscale', img)
    cv2.waitKey(0)

    edges = canny_edge_detector(img)
    edges = np.uint8(edges)
    cv2.imshow('Canny from Scratch', edges)
    cv2.imwrite("canny_scratch.png", edges)
    cv2.waitKey(0)

    # Compare to OpenCV Canny
    img = cv2.Canny(img, 50, 100)
    cv2.imshow('OpenCV Canny Edge Detector', img)
    cv2.imwrite("canny_opencv.png", img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()