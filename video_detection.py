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


def run_detection(video_path, detection_algorithm):
    capture = cv2.VideoCapture(video_path)

    if not capture.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    while True:
        ret, frame = capture.read()

        if not ret:
            print("End of video")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # edges = canny.canny_edge_detector(gray_frame)
        edges = opencv_prewitt(gray_frame)
        # edges = cv2.Canny(gray_frame, 50, 100)
        edges = np.uint8(edges)

        combined = cv2.hconcat([frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])

        cv2.imshow('Original and Edges', combined)

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