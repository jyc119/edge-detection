import numpy as np
import canny
import cv2
import time

# Run each algorithm on an image a number of times and average the time it takes
# Compare the algorithms times


def opencv_prewitt(img):
    img_gaussian = cv2.GaussianBlur(img, (5, 5), 0)

    # prewitt
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


def display_results(times):
    print("temp")

def main():
    img = cv2.imread('images/tomatoes.jpg', cv2.IMREAD_GRAYSCALE)
    averages = []
    for i in range(4):
        times = []
        if i == 0:
            for _ in range(100000):
                start = time.time_ns()
                img = cv2.GaussianBlur(img, (5, 5), 0)
                cv2.Canny(img, 50, 100)
                end = time.time_ns()
                times.append(end-start)
        if i == 1:
            start = time.time_ns()
            img = opencv_sobel(img)
            end = time.time_ns()
            times.append(end-start)
        if i == 2:
            start = time.time_ns()
            img = opencv_prewitt(img)
            end = time.time_ns()
            times.append(end-start)
        if i == 3:
            start = time.time_ns()
            img = cv2.GaussianBlur(img, (5, 5), 0)
            img = cv2.Laplacian(img, cv2.CV_64F)
            end = time.time_ns()
            times.append(end-start)

        avg = np.average(times)
        averages.append(avg)
    print(averages)
    display_results(averages)


if __name__ == "__main__":
    main()
