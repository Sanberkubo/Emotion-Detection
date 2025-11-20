import os
import cv2
import face_recognition
import numpy as np

class SimpleFacerec:
    def _init_(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        """
        Loads images from folder and extracts encodings.
        Saves ImageEncoding.npy and ImageNames.txt.
        """

        images = os.listdir(images_path)

        for image_name in images:
            image_path = os.path.join(images_path, image_name)
            img = cv2.imread(image_path)

            if img is None:
                print(f"[WARNING] Unable to read {image_name}")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_img)

            if len(encodings) > 0:
                encoding = encodings[0]
                self.known_face_encodings.append(encoding)

                # remove .jpg/.png/.jpeg
                name = os.path.splitext(image_name)[0]
                self.known_face_names.append(name)
            else:
                print(f"[WARNING] No face detected in {image_name}")

        # Save encodings
        np.save("ImageEncoding.npy", self.known_face_encodings)

        # Save names
        with open("ImageNames.txt", "w") as f:
            for name in self.known_face_names:
                f.write(name + "\n")

        print("[INFO] Face encodings created and saved successfully.")