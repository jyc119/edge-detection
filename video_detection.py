import cv2
import canny
import numpy as np


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

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame = cv2.resize(frame, (640, 360))

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # edges = canny.canny_edge_detector(gray_frame)
        edges = cv2.Canny(gray_frame, 50, 100)
        edges = np.uint8(edges)

        combined = cv2.hconcat([frame, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)])

        cv2.imshow('Original and Edges', combined)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

            # Release the capture and close windows
    capture.release()
    cv2.destroyAllWindows()


def main():
    video = 'videos/lane_test.mp4'
    run_detection(video, 'canny')


if __name__ == "__main__":
    main()