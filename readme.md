# 2D Object Recognition System

# Team Members

Mihir Chitre, Aditya Gurnani

## Project Description

This 2D Object Recognition project, authored by Mihir Chitre and Aditya Gurnani, is a comprehensive system that aims to identify and classify objects based on their shape, regardless of size, orientation, or position. Utilizing a top-down camera setup, this project achieves translation, scale, and rotation invariance in object recognition, allowing for real-time tracking and identification in a video stream or static images.

The system is built on several key tasks such as thresholding, morphological filtering, and segmentation to process and identify regions within images. After segmenting, it computes feature vectors for the central regions and compares them with a database of vectors, labeling each object by its closest match. It demonstrates real-time capabilities and uses confusion matrices for performance evaluation. The project also explores K-Nearest Neighbor and deep learning embeddings to further enhance accuracy and robustness.

## Requirements

- OpenCV library
- Webcam or video input source
- Objects for recognition

## Executables

There are no command line arguments for any executables in this project. All executables can be executed by simple double clicking them and further performing all image processing tasks on the live video stream.

- `threshold.exe` - Applies thresholding to separate objects from the background.
- `morphologicalFilter.exe` - Refines binary images.
- `segmentImages.exe` - Segments the image into regions written from scratch.
- `segmentImagesCV.exe` - Segments the image into regions.
- `computeFeatures.exe` - Computes features for each region.
- `trainingData.exe` - Collects feature vectors and labels.
- `classifyImages.exe` - Classifies objects using the nearest-neighbor approach.
- `classifyImagesKNN.exe` - Implements K-Nearest Neighbor classification.
- `classifyImagesDNN.exe` - Uses DNN for classification.
- `trainingDataDNN.exe` - Trains the system with DNN features.

## Usage

Steps to use the system:

1. Place the object on a white surface under the camera.
2. Run `threshold.exe` to start thresholding.
3. Use `morphologicalFilter.exe` for image cleanup.
4. Segment the image with `segmentImages.exe`.
5. Compute object features using `computeFeatures.exe`.
6. To collect training data, use `trainingData.exe` or `trainingDataDNN.exe`.
7. Classify objects with `classifyImages.exe`, `classifyImagesKNN.exe`, or `classifyImagesDNN.exe`.
8. Evaluate performance through the generated confusion matrices.

