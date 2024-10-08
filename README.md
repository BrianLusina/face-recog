# FaceRecog

This is a simple Python CLI program that is trained to learn to recognize and detect faces.

Note this program works well for training on images that contain a single face. If you want to train on images with multiple identifiable faces, then feel free to add an alternative strategy for marking the faces in the training images for the program to detect.

## Pre-requisites

Before you start installing this project’s dependencies with [pip](https://pip.pypa.io/en/stable/) or [uv](https://github.com/astral-sh/uv) or [pipenv](https://pipenv.pypa.io/en/latest/) or [poetry](https://python-poetry.org/), you’ll need to ensure that you have [CMake](https://cmake.org/) and a C compiler like [gcc](https://gcc.gnu.org/) installed on your system. If your system doesn’t already have them installed, then follow these instructions to get started:

### Windows

To install CMake on Windows, visit the [CMake downloads page](https://cmake.org/download/) and install the appropriate installer for your system.

You can’t get gcc as a stand-alone download for Windows, but you can install it as a part of the [MinGW](https://www.mingw-w64.org/) runtime environment through the [Chocolatey package manager](https://chocolatey.org/) with the following command:

```shell
PS> choco install mingw
```

> Windows PowerShell

### Linux

To install CMake on Linux, visit the [CMake downloads page](https://cmake.org/download/) and install the appropriate installer for your system. Alternatively, CMake binaries may also be available through your favorite package manager. If you use [apt package management](https://ubuntu.com/server/docs/package-management), for example, then you can install CMake with this:

```shell
sudo apt-get update
sudo apt-get install cmake
```

> Linux Shell, e.g. fish, zsh, bash, etc

You’ll also install gcc through your package manager. To install gcc with apt, you’ll install the build-essential metapackage:

```shell
sudo apt-get install build-essential
```

To verify that you’ve successfully installed gcc, you can check the version:

``` shell
$ gcc --version
gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE
```

If this returns a version number, then you’re good to go!

### MacOS

To install CMake on macOS, visit the [CMake downloads page](https://cmake.org/download/) and install the appropriate installer for your system. If you have [Homebrew](https://brew.sh/) installed, then you can install both CMake and gcc that way:

```shell
brew update
brew install cmake gcc
```

After following these steps for your operating system, you’ll have Cmake and gcc installed and ready to assist you in building the project.

---

Next step is to install the dependencies.

Depending on the package manager being used. Here [uv](https://github.com/astral-sh/uv) is being used, the steps will be different for installing the dependencies. However, the dependencies are outlined in the [pyproject.toml](./pyproject.toml) file. Follow the steps of the package manager used to install dependencies required.

Note that [uv](https://docs.astral.sh/uv/) is the package manager of choice, therefore it is recommended that it is used for this project.

## Project structure

The main source code can be found in the [src](./src/) folder.

``` plain
face-recog/
│
├── output/
│   └── .gitkeep
├── training/
│   └── ben_affleck/
│       ├── img_1.jpg
│       └── img_2.png
│   └── .gitkeep
├── validation/
│   ├── ben_affleck1.jpg
│   └── michael_jordan1.jpg
├── src/
│   └── face/
|      └── __main__.py
│      └── ... other files
└── ...other files
```

> A simple directory structure of the project

Note that the sample provided above may not be in line with the current repository.

### Validation folder

You can place the validation images directly into the validation/ directory. Your validation images need to be images that you don’t train with, but you can identify the people who appear in them.

### Training folder

For training/, you should have images separated by subject into directories with the subject’s name as shown in the example above. Setting the training directory up this way will allow giving the face recognizer the information that it needs to associate a label—the person pictured—with the underlying image data.

## Technology used

- [Face Recognition](https://github.com/ageitgey/face_recognition) - Facial Recognition API for Python
- [Python](https://www.python.org/) - Programming Language
- [Numpy](https://numpy.org/) - Scientific Computing Python Package

