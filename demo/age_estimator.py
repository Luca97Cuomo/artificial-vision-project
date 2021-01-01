import tensorflow
import numpy
import cv2
import argparse
from pathlib import Path


def estimate_age(model_path):

    # Load the model

    # Execute face detection and resizing for the net


    cam = cv2.VideoCapture(0)

    cv2.namedWindow("Age estimator")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed capturing a frame")
            break
        cv2.imshow("Age estimator", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Bye :-)")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = f"opencv_frame_{img_counter}.png"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1

    cam.release()

    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Estimates your age.')
    parser.add_argument('-m', '--model_path', type=str, help='The path of the neural network you want to use for age estimation',
                        required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path).resolve()

    estimate_age(str(model_path))


if __name__ == '__main__':
    main()
