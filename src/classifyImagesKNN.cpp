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
#include "kmeans.h"
#include "objectRecogFunctions.h"
#include <corecrt_math_defines.h>

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

    while (true)
    {
        cap >> frame; 
        if (frame.empty())
            break; 

        cv::Mat cleaned = applyMorphologicalFilter(frame); 

        classifyAndLabelRegionsKNN(cleaned, output, frame, minRegionSize); 

        cv::imshow("Output", frame); 
        if (cv::waitKey(30) >= 0)
            break; 
    }
    return 0;
}