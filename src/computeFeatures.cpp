// computeFeatures.cpp
// Author: Mihir Chitre, Aditya Gurnani
// Date: 02/24/2024
// Description: This program captures video from a webcam, applies morphological filtering, segments image into different regions, finds the main or the
//              biggest region in the frame and computes its features. These features are displayed on the frame with bounding box and axis of least 
//              central movement on the object.

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

        findRegions(cleaned, output, frame, minRegionSize); 

        cv::imshow("Output", output); 
        if (cv::waitKey(30) >= 0)
            break; 
    }
    return 0;
}