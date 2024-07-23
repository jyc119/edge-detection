import cv2
import canny
import sobel
import prewitt
import numpy as np


def define_region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, (255, 0, 0))
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_lane_lines(img, lines):
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 5)


def run_detection(video_path):
    capture = cv2.VideoCapture(video_path)

    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            print("End of video")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)

        # Our implementation is too slow for some reason
        # edges = canny.canny_edge_detector(blur)
        edges = cv2.Canny(blur, 50, 100)
        edges = np.uint8(edges)

        h, w = edges.shape
        vertices = np.array([[(0, h), (w/2, 2*h/3), (w, h)]], np.int32)
        roi = define_region_of_interest(edges, vertices)

        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=50,
                                maxLineGap=150)
        line_image = np.zeros_like(frame)
        draw_lane_lines(line_image, lines)

        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0.0)

        cv2.imshow('Lane Lines Using Canny Edge Detection', combo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def main():
    video = 'videos/lane_test_2.mp4'
    run_detection(video,)


if __name__ == "__main__":
    main()
