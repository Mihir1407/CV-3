// classifyImages.cpp
// Author: Mihir Chitre, Aditya Gurnani
// Date: 02/24/2024
// Description: This program captures video from a webcam, processes each frame to identify regions, computes the feature vectors for biggest central region
//              and compares the computed feature vector with the feature vectors from vectors database file. Then it attaches the object in frame with the
//              label of feature vector which closely matches with the computed vector. This follows a nearest-neighbor approach.

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

std::vector<std::vector<int>> confusionMatrix(6, std::vector<int>(6, 0));

/*
   Function: printConfusionMatrix
   Purpose: Print the confusion matrix.
   Arguments:
       matrix (const std::vector<std::vector<int>>&) : Confusion matrix to be printed.
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

    // Print matrix rows with row labels
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
int main()
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
    {
        std::cerr << "Error opening video capture" << std::endl;
        return -1;
    }

    cv::Mat frame, output;
    int minRegionSize = 500;
    std::vector<std::string> labels = {"cup", "controller", "slipper", "glasses", "plate", "key"};
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        cv::Mat cleaned = applyMorphologicalFilter(frame);
        std::string identifiedLabel = classifyAndLabelRegions(cleaned, output, frame, minRegionSize);

        cv::imshow("Output", frame);
        int key = cv::waitKey(30);
        if (key == 'n')
        {

            std::string trueLabel;
            std::cout << "Enter true label: ";
            std::cin >> trueLabel;
            std::cout << "True Label: " << trueLabel << ", Identified Label: " << identifiedLabel << std::endl;
            // Update the confusion matrix
            if (labelToIndex.find(trueLabel) != labelToIndex.end() && labelToIndex.find(identifiedLabel) != labelToIndex.end())
            {
                int trueIndex = labelToIndex[trueLabel];
                int identifiedIndex = labelToIndex[identifiedLabel];
                std::cout << "Updating Matrix Cell[" << trueIndex << "][" << identifiedIndex << "]" << std::endl;
                confusionMatrix[trueIndex][identifiedIndex]++;
            }
            else
            {
                std::cerr << "Invalid label entered or identified." << std::endl;
            }

            printConfusionMatrix(confusionMatrix, labels);
        }
        else if (key >= 0)
        {
            break;
        }
    }
    return 0;
}