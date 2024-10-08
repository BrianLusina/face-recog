from pathlib import Path
import pickle

import face_recognition

from .models import ModelType

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")

# Creates the 3 directories if they do not already exist
Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)


def encode_known_faces(
    model: ModelType = ModelType.HOG, encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    """Encodes known faces taking in a model type and an encoding location path. This goes through
    every directory in `training` folder, saving the label of each directory into name and the encodings
    as well.

    Args:
        model (ModelType, optional): ModelType to use for the face recognition algorithm. Defaults to "hog".
        encodings_location (Path, optional): Encodings location path. Defaults to DEFAULT_ENCODINGS_PATH.
    """
    names = []
    encodings = []
    for filepath in Path("training").glob("*/*"):
        name = filepath.parent.name
        image = face_recognition.load_image_file(filepath)

        # this returns a list of four-element tuples, one tuple for each detected face.
        # The four elements per tuple provide the four coordinates of a box that could surround the detected face.
        # Such a box is also known as a bounding box.
        face_locations = face_recognition.face_locations(img=image, model=model.name)

        # This is used to generate encodings for the detected faces in an image.
        # An encoding is a numeric representation of facial features that’s used to match similar faces by their features.
        face_encodings = face_recognition.face_encodings(
            face_image=image, known_face_locations=face_locations
        )

        for encoding in face_encodings:
            names.append(name)
            encodings.append(encoding)

    # The names and encodings are combined into a dictionary and saved to disk. This is because
    # generating encodings can be time-consuming, especially if you don’t have a dedicated GPU. 
    # Once they’re generated, saving them allows you to reuse the encodings in other parts of your code without re-creating 
    # them every time.
    name_encodings = dict(names=names, encodings=encodings)
    with encodings_location.open("wb") as f:
        pickle.dump(name_encodings, f)
