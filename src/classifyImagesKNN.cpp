// classifyImagesKNN.cpp
// Author: Mihir Chitre, Aditya Gurnani
// Date: 02/24/2024
// Description: This program captures video from a webcam, processes each frame to identify regions, computes the feature vectors for biggest central region
//              and compares the computed feature vector with the feature vectors from vectors database file. Then it attaches the object in frame with the 
//              label of the object, whose K-nearest neighbors have the least average distance from the computed feature vector.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include <iomanip> 
#include "kmeans.h"
#include "objectRecogFunctions.h"
#include <corecrt_math_defines.h>

// Map labels to their corresponding indices
std::map<std::string, int> labelToIndex = {
    {"cup", 0},
    {"controller", 1},
    {"slipper", 2},
    {"glasses", 3},
    {"plate", 4},
    {"key", 5}};

// Confusion matrix to track the performance of the classification.
std::vector<std::vector<int>> confusionMatrix(6, std::vector<int>(6, 0));

/*
   Function: printConfusionMatrix
   Purpose: Prints the confusion matrix to the console. 
   Arguments:
       matrix (const std::vector<std::vector<int>>&): The confusion matrix to be printed.
       labels (const std::vector<std::string>&): A list of labels corresponding to the indices of the confusion matrix.
   Returns: None
*/
void printConfusionMatrix(const std::vector<std::vector<int>>& matrix, const std::vector<std::string>& labels) {
    std::cout << "Confusion Matrix:\n";
    std::cout << std::left << std::setw(12) << " "; 

    // Print column labels
    for (const auto& label : labels) {
        std::cout << std::setw(12) << label;
    }
    std::cout << "\n";
    for (int i = 0; i < labels.size(); i++) {
        std::cout << std::setw(12) << labels[i]; 
        for (int val : matrix[i]) {
            std::cout << std::setw(12) << val; 
        }
        std::cout << "\n";
    }
}

/*
   Function: main
   Purpose: Entry point of the program.
   Returns: 0 on successful execution, -1 otherwise.
*/
int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video capture" << std::endl;
        return -1;
    }

    cv::Mat frame, output;
    int minRegionSize = 500;
    std::vector<std::string> labels = {"cup", "controller", "slipper", "glasses", "plate", "key"};
    
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat cleaned = applyMorphologicalFilter(frame);
        std::string predictedLabel = classifyAndLabelRegionsKNN(cleaned, output, frame, minRegionSize);

        cv::imshow("Output", frame);
        int key = cv::waitKey(30);
        if (key == 'n') {
            std::string trueLabel;
            std::cout << "Enter true label: ";
            std::cin >> trueLabel;

            if (labelToIndex.find(trueLabel) != labelToIndex.end() && labelToIndex.find(predictedLabel) != labelToIndex.end()) {
                int trueIndex = labelToIndex[trueLabel];
                int predictedIndex = labelToIndex[predictedLabel];
                confusionMatrix[trueIndex][predictedIndex]++;
            } else {
                std::cerr << "Invalid label entered or identified." << std::endl;
            }

            printConfusionMatrix(confusionMatrix, labels);
        } else if (key >= 0) {
            break;
        }
    }
    return 0;
}
