from .detector import encode_known_faces, recognize_faces, validate
from .cli import get_command_line_args
from .models import ModelType


def main() -> None:
    args = get_command_line_args()

    if args.train:
        encode_known_faces()

    if args.validate:
        validate(model=ModelType(args.model))

    if args.test:
        recognize_faces(image_location=args.f, model=ModelType(args.model))


if __name__ == "__main__":
    main()
