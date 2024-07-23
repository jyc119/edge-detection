import numpy as np

# Run each algorithm on an image a number of times and average the time it takes
# Compare the algorithms times


def main():
    averages = []
    for i in range(4):
        times = []
        if i == 0:
            print("Canny OpenCV")
        if i == 1:
            print("Sobel OpenCV")
        if i == 2:
            print("Prewitt")
        if i == 3:
            print("Laplacian")

        avg = np.average(times)
        averages.append(avg)


if __name__ == "__main__":
    main()
