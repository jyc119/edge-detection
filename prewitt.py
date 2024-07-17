import numpy as np
import cv2
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc


def evaluate_metrics(egt, detected_edges):
    # Ensure the images are binary
    egt = (egt > 128).astype(np.uint8)
    detected_edges = (detected_edges > 128).astype(np.uint8)

    # Flatten the arrays to calculate the metrics
    egt_flat = egt.flatten()
    detected_edges_flat = detected_edges.flatten()

    # Calculate Precision, Recall, and F1 Score
    precision = precision_score(egt_flat, detected_edges_flat)
    recall = recall_score(egt_flat, detected_edges_flat)
    f1 = f1_score(egt_flat, detected_edges_flat)

    # Calculate ROC AUC
    fpr, tpr, _ = roc_curve(egt_flat, detected_edges_flat)
    roc_auc = auc(fpr, tpr)

    return precision, recall, f1, roc_auc

def opencv_prewitt():

    img = cv2.imread('images/tomatoes.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gaussian = cv2.GaussianBlur(gray, (3, 3), 0)

    # prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    return img_prewittx + img_prewitty

def gaussian(img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imshow("Gaussian Blur", img_gaussian)
    cv2.waitKey(0)
    return img_gaussian

def prewitt(img):
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

    # Apply Gaussian blur
    img_gaussian = gaussian(gray)

    # Apply Prewitt
    edges_prewitt = prewitt(img_gaussian)
    ground_truth = opencv_prewitt()
    precision, recall, f1, roc_auc = evaluate_metrics(ground_truth, edges_prewitt)
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
