import numpy as np
import cv2


def gaussian(img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("Gaussian kernel", img_gaussian)
    cv2.waitKey(0)
    return img_gaussian


def laplacian(img):
    # Define Laplacian kernel (3x3)
    kernel = np.array([[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]])

    # Convolve with the Laplacian kernel
    edges = cv2.filter2D(img, -1, kernel)

    # Taking absolute value of the result
    edges = np.abs(edges)
    edges = np.clip(edges, 0, 255)  # Clip values to stay within byte range
    edges = edges.astype(np.uint8)  # Convert to unsigned 8-bit integer

    cv2.imshow("Laplacian Edge Detection", edges)
    cv2.waitKey(0)

    return edges


def main():
    print("Laplacian Edge Detection Test")

    img = cv2.imread('images/tomatoes.jpg')
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)

    # Apply Gaussian blur
    img_gaussian = gaussian(gray)

    # Apply Laplacian
    edges_laplacian = laplacian(img_gaussian)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
