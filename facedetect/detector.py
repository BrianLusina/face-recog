from typing import Optional, Dict, List, Tuple, Any, Literal
from pathlib import Path
from collections import Counter
import pickle

import face_recognition
from numpy.typing import NDArray
from PIL import Image, ImageDraw

from .models import ModelType

DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

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


def recognize_faces(
    image_location: str,
    model: ModelType = ModelType.HOG,
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    """Recognizes faces in images that don't have a label

    Args:
        image_location (str): image location
        model (ModelType): Model type to use, defaults to ModelType.HOG
        encoding_location (Path, optional): encoding location. Defaults to DEFAULT_ENCODINGS_PATH.
    """

    with encodings_location.open("rb") as f:
        loaded_encodings = pickle.load(f)

    input_image = face_recognition.load_image_file(image_location)

    input_face_locations = face_recognition.face_locations(input_image, model=model)

    input_face_encodings = face_recognition.face_encodings(
        face_image=input_image, known_face_locations=input_face_locations
    )

    pillow_image = Image.fromarray(input_image)
    # this will help with drawing a bounding box around the detected image
    draw = ImageDraw.Draw(pillow_image)

    # Now we use the encoding of the detected face to make a comparison with all of the encodings that were found in the previous step.
    # This will happen within a loop so that we can detect and recognize multiple faces in the unknown image

    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)

    del draw
    pillow_image.show()


def _recognize_face(
    unknown_encoding: NDArray, loaded_encodings: Dict[str, List[NDArray]]
) -> Optional[NDArray]:
    """Recognize an unknown encoding of a face and compare it to the previous loaded encodings
    If a face matches, then it is returned. If not, None is returned

    Args:
        unknown_encoding (NDArray): Encoding of an unmatched face
        loaded_encodings (Dict[str, List[NDArray]]): Loaded encodings of previously matched faces

    Returns:
        Optional[NDArray]: encoding of matched face
    """
    boolean_matches = face_recognition.compare_faces(
        known_face_encodings=loaded_encodings["encodings"],
        face_encoding_to_check=unknown_encoding,
    )
    votes = Counter(
        name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match
    )
    if votes:
        return votes.most_common(1)[0][0]


def _display_face(
    draw: ImageDraw,
    bounding_box: Tuple[int, Any, Any, int],
    name: NDArray | Literal["Unknown"],
) -> None:
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox((left, bottom), name)
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )
