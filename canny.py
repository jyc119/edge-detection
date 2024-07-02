import numpy as np
import cv2 as cv


def canny_edge_detector(img):
    img = cv.imread(img, 0)
    return img


def main():
    print("Canny Edge Detection Test")
    img = canny_edge_detector('test_image.png')
    cv.imshow('Canny Edge Detector', img)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
