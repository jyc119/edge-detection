import numpy as np
import cv2


def gaussian(img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("Gaussian kernel", img_gaussian)
    cv2.waitKey(0)

    return img_gaussian


def sobel(img):
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

    # Convolve with the Sobel kernels
    edges_x = cv2.filter2D(img, -1, Gx)
    edges_y = cv2.filter2D(img, -1, Gy)

    # Calculate the magnitude of the gradients
    magnitude = np.sqrt(edges_x ** 2 + edges_y ** 2)
    magnitude = np.clip(magnitude, 0, 255)

    cv2.imshow("Sobel X", edges_x)
    cv2.waitKey(0)
    cv2.imshow("Sobel Y", edges_y)
    cv2.waitKey(0)
    cv2.imshow("Sobel Edge Magnitude", magnitude.astype(np.uint8))
    cv2.waitKey(0)

    return magnitude


def main():
    print("Sobel Edge Detection Test")

    img = cv2.imread('images/tomatoes.jpg')
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)

    # Apply gaussian blur
    img_gaussian = gaussian(gray)
    v = sobel(img_gaussian)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
