import numpy as np
import canny
import cv2
import time

# Run each algorithm on an image a number of times and average the time it takes
# Compare the algorithms times


def display_results(times):
    print("temp")

def main():
    img = cv2.imread('images/tomatoes.jpg', cv2.IMREAD_GRAYSCALE)
    averages = []
    for i in range(4):
        times = []
        if i == 0:
            for _ in range(100):
                start = time.time_ns()
                img = cv2.GaussianBlur(img, (5, 5), 0)
                cv2.Canny(img, 50, 100)
                end = time.time_ns()
                times.append(end-start)
        if i == 1:
            print("Sobel OpenCV")
            times.append(1)
        if i == 2:
            print("Prewitt")
            times.append(1)
        if i == 3:
            print("Laplacian")
            times.append(1)

        avg = np.average(times)
        averages.append(avg)
    print(averages)
    display_results(averages)


if __name__ == "__main__":
    main()
