import numpy as np
import cv2

def gaussian(img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("Gaussian kernel", img_gaussian)
    cv2.waitKey(0)

    return img_gaussian

def main():
    print("Laplacian Edge Detection Test")

    img = cv2.imread('images/tomatoes.jpg')
    cv2.imshow("Original Image", img)
    cv2.waitKey(0)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Grayscale Image", gray)
    cv2.waitKey(0)

    # Apply gaussian blur
    img_gaussian = gaussian(gray)
    # v = sobel(img_gaussian)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
