"""Defines CLI arguments"""

from argparse import Namespace, ArgumentParser
from .models import ModelType


def get_command_line_args() -> Namespace:
    """Creates a parser and adds arguments for the parser returning the namespace for the argument parser to use in a CLI
    Returns:
        Namespace: populated namespace object
    """
    parser = ArgumentParser(
        prog="face-recog", description="Recognize faces in an image"
    )

    parser.add_argument("--train", action="store_true", help="Train on input data")

    parser.add_argument(
        "--validate", action="store_true", help="Validate trained model"
    )

    parser.add_argument(
        "--test", action="store_true", help="Test the model with an unknown image"
    )

    parser.add_argument(
        "-m",
        "--model",
        action="store",
        default=ModelType.HOG.name,
        choices=[ModelType.HOG.name, ModelType.CNN.name],
        help="Which model to use for training: hog(CPU), cnn(GPU)"
    )
    
    parser.add_argument("-f", action="store", help="Path to an image with an unknown face")
    
    return parser.parse_args()