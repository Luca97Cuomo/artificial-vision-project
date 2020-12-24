from imutils.face_utils import FaceAligner
import dlib
import cv2


def pose_normalization_image(shape_predictor, image_path, new_image_path, verbose=False):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(shape_predictor)
    face_aligner = FaceAligner(predictor, desiredFaceWidth=192, desiredLeftEye=(0.30, 0.30))

    # load image
    image = cv2.imread(image_path)

    # convert it to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    aligned_image = None
    rects = detector(gray, 2)
    if rects is not None and len(rects) > 0:
      # I am interested only in the first image of the dataset
      rect = rects[0]
      # align the face using facial landmarks
      aligned_image = face_aligner.align(image, gray, rect)

    if aligned_image is None:
        if verbose:
            print("Unable to align the image. Saving the original image")
        cv2.imwrite(new_image_path, image)
    else:
        if verbose:
            print("Successfully aligned the image. Saving the aligned image")
        cv2.imwrite(new_image_path, aligned_image)
