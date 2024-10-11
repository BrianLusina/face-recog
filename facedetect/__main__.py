from .detector import encode_known_faces, recognize_faces

def main() -> None:
    encode_known_faces()
    recognize_faces("unknown.jpg")


if __name__ == "__main__":
    main()
