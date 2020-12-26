from preprocessing import pose_normalization_image
import os

pose_normalization_image(os.path.join("..", "shape_predictor_68_face_landmarks.dat"), os.path.join("..", "test_images", "original", "emma_2.jpg"), os.path.join("..", "test_images", "aligned", "emma_2_aligned.jpg"),
                         True)