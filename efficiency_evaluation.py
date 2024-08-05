import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import canny


def opencv_prewitt(img):
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    return img_prewittx + img_prewitty


def opencv_sobel(g):
    g = cv2.GaussianBlur(g, (5, 5), 0)

    img_sobelx = cv2.Sobel(g, cv2.CV_8U, 1, 0, ksize=5)
    img_sobely = cv2.Sobel(g, cv2.CV_8U, 0, 1, ksize=5)
    img_sobel = img_sobelx + img_sobely

    return img_sobel


def canny_scratch(gaussian_img):
    G, theta = canny.gradient_intesity(gaussian_img)

    # Apply magnitude thresholding
    Z = canny.magnitude_thresholding(G, theta)

    # Apply double threshold
    threshold_img = canny.double_threshold(Z, 0.05, 0.1)

    # Track edge by hysteresis
    final_img = canny.hysteresis(threshold_img)
    return final_img


def display_results(times, name):
    for i in range(len(times)):
        times[i] = times[i] * 1000
    labels = ['Canny', 'Sobel', 'Prewitt', 'Laplacian']
    left = [1, 2, 3, 4]

    plt.bar(left, times, tick_label=labels, width=0.8)

    plt.xlabel('Edge Detection Algorithm')
    plt.ylabel('Runtime (ms)')
    plt.title('Average Runtime to Detect Edges on Image')

    plt.savefig(name)
    plt.show()


def display_results_scratch(times, name):
    for i in range(len(times)):
        times[i] = times[i] * 1000
    labels = ['Sobel', 'Prewitt', 'Laplacian']
    left = [1, 2, 3]

    plt.bar(left, times[1:], tick_label=labels, width=0.8)

    plt.xlabel('Edge Detection Algorithm')
    plt.ylabel('Runtime (ms)')
    plt.title('Average Runtime to Detect Edges on Image')

    plt.savefig(name)
    plt.show()


def sobel_scratch(img):
    # Define Sobel kernels
    Gx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]], dtype=np.float32)

    Gy = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]], dtype=np.float32)

    # Convolve with the Sobel kernels
    edges_x = cv2.filter2D(img, cv2.CV_32F, Gx)
    edges_y = cv2.filter2D(img, cv2.CV_32F, Gy)

    # Calculate the magnitude of the gradients
    magnitude = cv2.magnitude(edges_x, edges_y)

    # Normalize the magnitude to the range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)  # Convert to unsigned 8-bit integer

    return magnitude


def prewitt_scratch(img):
    # Define Prewitt kernels
    Gx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]], dtype=np.float32)

    Gy = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]], dtype=np.float32)

    # Convolve with the Prewitt kernels
    edges_x = cv2.filter2D(img, cv2.CV_32F, Gx)
    edges_y = cv2.filter2D(img, cv2.CV_32F, Gy)

    # Calculate the magnitude of the gradients
    magnitude = cv2.magnitude(edges_x, edges_y)

    # Normalize the magnitude to the range [0, 255]
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    magnitude = magnitude.astype(np.uint8)  # Convert to unsigned 8-bit integer

    return magnitude


def laplacian_scratch(img):
    # Define Laplacian kernel (3x3)
    kernel = np.array([[1, 4, 1],
                       [4, -20, 4],
                       [1, 4, 1]])

    # Convolve with the Laplacian kernel
    edges = cv2.filter2D(img, cv2.CV_32F, kernel)

    # Taking absolute value of the result
    edges = np.abs(edges)
    # Normalize the result to the range [0, 255]
    edges = cv2.normalize(edges, None, 0, 255, cv2.NORM_MINMAX)
    edges = edges.astype(np.uint8)  # Convert to unsigned 8-bit integer

    return edges


def scratch_evaluation():
    img = cv2.imread('images/tomatoes.jpg', cv2.IMREAD_GRAYSCALE)
    averages = []
    for i in range(4):
        times = []
        if i == 0:
            for _ in range(1):
                start = time.time()
                img = cv2.GaussianBlur(img, (5, 5), 0)
                canny_scratch(img)
                end = time.time()
                times.append(end - start)
        if i == 1:
            for _ in range(100000):
                start = time.time()
                img = cv2.GaussianBlur(img, (5, 5), 0)
                sobel_scratch(img)
                end = time.time()
                times.append(end - start)
        if i == 2:
            for _ in range(100000):
                start = time.time()
                img = cv2.GaussianBlur(img, (5, 5), 0)
                prewitt_scratch(img)
                end = time.time()
                times.append(end - start)
        if i == 3:
            for _ in range(100000):
                start = time.time()
                img = cv2.GaussianBlur(img, (5, 5), 0)
                laplacian_scratch(img)
                end = time.time()
                times.append(end - start)

        avg = np.average(times)
        averages.append(avg)
    print(averages)
    display_results_scratch(averages, 'images/scratch_efficiency_evaluation.png')


def main():
    img = cv2.imread('images/tomatoes.jpg', cv2.IMREAD_GRAYSCALE)
    averages = []
    for i in range(4):
        times = []
        if i == 0:
            for _ in range(10000):
                start = time.time()
                img = cv2.GaussianBlur(img, (5, 5), 0)
                cv2.Canny(img, 50, 100)
                end = time.time()
                times.append(end-start)
        if i == 1:
            for _ in range(10000):
                start = time.time()
                img = opencv_sobel(img)
                end = time.time()
                times.append(end-start)
        if i == 2:
            for _ in range(10000):
                start = time.time()
                img = opencv_prewitt(img)
                end = time.time()
                times.append(end-start)
        if i == 3:
            for _ in range(10000):
                start = time.time()
                img = cv2.GaussianBlur(img, (5, 5), 0)
                img = cv2.Laplacian(img, cv2.CV_64F)
                end = time.time()
                times.append(end-start)

        avg = np.average(times)
        averages.append(avg)
    print(averages)
    display_results(averages, 'images/efficiency_evaluation.png')


if __name__ == "__main__":
    # main()
    scratch_evaluation()