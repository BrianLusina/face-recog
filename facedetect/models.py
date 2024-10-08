"""
Defines the models to be used
"""

from enum import Enum


class ModelType(Enum):
    """Model Type to be used by the face recognition algorithm. There are only 2 types:
    
    - HOG(hog) which is HOG (histogram of oriented gradients) - is a common technique for object detection. 
    Which Works best with a CPU. More about this here https://www.pyimagesearch.com/2014/11/10/histogram-oriented-gradients-object-detection/
    
    - CNN (convolutional neural network) is another technique for object detection. In contrast to a HOG, a CNN works better 
    on a GPU, otherwise known as a video card.

    Args:
        Enum (str): Model to use
    """

    HOG = "hog"
    CNN = "CNN"
