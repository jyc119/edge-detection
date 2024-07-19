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


def run_detection(video_path, detection_algorithm):
    capture = cv2.VideoCapture(video_path)

    while capture.isOpened():
        ret, frame = capture.read()

        if not ret:
            print("End of video")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        # edges = canny.canny_edge_detector(gray_frame)
        # edges = opencv_prewitt(gray_frame)
        edges = cv2.Canny(blur, 50, 100)
        # edges = np.uint8(edges)

        h, w = edges.shape
        vertices = [(0, h), (w/2, 2*h/3), (w, h)]
        roi = define_region_of_interest(edges, np.array([vertices], np.int32))

        # combined = cv2.hconcat([frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])

        cv2.imshow('Original and Edges', roi)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

            # Release the capture and close windows
    capture.release()
    cv2.destroyAllWindows()


def main():
    video = 'videos/lane_test.mp4'
    run_detection(video, 'canny')


if __name__ == "__main__":
    main()