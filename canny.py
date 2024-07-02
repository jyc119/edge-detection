import numpy as np
import cv2 as cv


# def gaussian_filter(img):



def canny_edge_detector(img):
    # smooth_img = gaussian_filter(img)

    # Find intensity gradients

    # Apply magnitude thresholding

    # Apply double threshold

    # Track edge by hysteresis
    final_img = img
    return final_img


def main():
    print("Canny Edge Detection Test")

    # Convert to grayscale
    img = cv.imread('test_image_1.png', 0)
    img = canny_edge_detector(img)

    # Compare to OpenCV Canny
    # img = cv.Canny(img, 100, 200)
    cv.imshow('Canny Edge Detector', img)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
