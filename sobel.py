import numpy as np
import cv2
from evaluation import evaluate_metrics

def opencv_sobel(g):
    # Sobel
    img_sobelx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=5)
    img_sobely = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = cv2.magnitude(img_sobelx, img_sobely)
    sobel = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return sobel

def gaussian(img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("Gaussian Blur", img_gaussian)
    cv2.waitKey(0)
    return img_gaussian

def sobel(img):
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

    cv2.imshow("Sobel Edge Magnitude", magnitude)
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

    # Apply Gaussian blur
    img_gaussian = gaussian(gray)

    # Apply Sobel
    edges_sobel = sobel(img_gaussian)
    cv2.imwrite("sobel_scratch.png", edges_sobel)
    ground_truth = opencv_sobel(img_gaussian)
    cv2.imwrite("sobel_opencv.png", ground_truth)
    precision, recall, f1, roc_auc = evaluate_metrics(ground_truth, edges_sobel)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
