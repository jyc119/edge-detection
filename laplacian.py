import numpy as np
import cv2
from evaluation import evaluate_metrics

def opencv_laplacian(img):
    # Apply the Laplacian operator
    laplacian = cv2.Laplacian(img, cv2.CV_64F)

    # Take the absolute value of the Laplacian to avoid negative values
    laplacian = np.abs(laplacian)

    # Normalize the result to the range [0, 255]
    laplacian = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX)

    # Convert to unsigned 8-bit integer format
    laplacian = laplacian.astype(np.uint8)

    cv2.imshow("Laplacian Edge Detection", laplacian)
    cv2.waitKey(0)

    return laplacian

def gaussian(img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("Gaussian kernel", img_gaussian)
    cv2.waitKey(0)
    return img_gaussian


def laplacian(img):
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
    cv2.imwrite("images/laplacian_scratch.png", edges_laplacian)
    ground_truth = opencv_laplacian(img_gaussian)
    cv2.imwrite("images/laplacian_opencv.png", ground_truth)
    precision, recall, f1, roc_auc = evaluate_metrics(ground_truth, edges_laplacian)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
