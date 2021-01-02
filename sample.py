import copy
import cv2 as cv

from polygon_filter import polygon_filter


def main():
    cap = cv.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv.resize(frame, (960, 540))
        original_frame = copy.deepcopy(frame)

        frame = polygon_filter(frame)

        cv.imshow('original', original_frame)
        cv.imshow('polygon filter', frame)
        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()