import numpy as np
import cv2

def gaussian(img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("Gaussian kernel", img_gaussian)
    cv2.waitKey(0)

    return img_gaussian

def prewitt(img):
    # Define Prewitt kernels
    Gx = np.array([[1, 0, -1],
                   [1, 0, -1],
                   [1, 0, -1]])

    Gy = np.array([[1, 1, 1],
                   [0, 0, 0],
                   [-1, -1, -1]])

    # Convolve with the Prewitt kernels
    edges_x = cv2.filter2D(img, -1, Gx)
    edges_y = cv2.filter2D(img, -1, Gy)

    # Calculate the magnitude of the gradients
    magnitude = np.sqrt(edges_x**2 + edges_y**2)
    magnitude = np.clip(magnitude, 0, 255)  # Clip values to stay within byte range

    magnitude = magnitude.astype(np.uint8)  # Convert to unsigned 8-bit integer

    cv2.imshow("Prewitt Edge Magnitude", magnitude)
    cv2.waitKey(0)

    return magnitude

def main():
    print("Prewitt Edge Detection Test")

    img = cv2.imread('images/tomatoes.jpg')
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)

    # Apply gaussian blur
    img_gaussian = gaussian(gray)

    # Apply Prewitt
    edges_prewitt = prewitt(img_gaussian)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
