import cv2
import canny
import sobel
import prewitt
import numpy as np

def opencv_prewitt(img):
    img_gaussian = cv2.GaussianBlur(img, (3, 3), 0)

    # prewitt
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    img_prewittx = cv2.filter2D(img_gaussian, -1, kernelx)
    img_prewitty = cv2.filter2D(img_gaussian, -1, kernely)

    return img_prewittx + img_prewitty


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


def run_detection(video_path, detection_algorithm):
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
        vertices = [(0, h), (w/2, 2*h/3), (w, h)]
        roi = define_region_of_interest(edges, np.array([vertices], np.int32))

        lines = cv2.HoughLinesP(roi, 1, np.pi / 180, 50, minLineLength=50,
                                maxLineGap=150)
        line_image = np.zeros_like(frame)
        draw_lane_lines(line_image, lines)

        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 0.0)

        combined = cv2.hconcat([frame, cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)])

        cv2.imshow('Lane Lines', combo_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()


def main():
    video = 'videos/lane_test_2.mp4'
    run_detection(video, 'canny')


if __name__ == "__main__":
    main()