// segmentImages.cpp
// Author: Mihir Chitre, Aditya Gurnani
// Date: 02/24/2024
// Description: This program captures video from a webcam, applies morphological filtering, segments the image into multiple colored regions
//              and displays region maps in a window. This program uses a custom implementation of two-pass connected components algorithm
//              to segment the image.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include "kmeans.h"
#include "objectRecogFunctions.h"

/*
   Function: main
   Purpose: Entry point of the program.
   Returns: 0 on successful execution, -1 otherwise.
*/
int main() {

    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    while(true) {
        cv::Mat frame, processedFrame;
        cap >> frame; 
        
        if (frame.empty()) 
            break;

        processedFrame = applyConnectedComponentsAndDisplayRegions(frame);

        cv::imshow("Region Maps", processedFrame);

        if (cv::waitKey(30) >= 0) 
            break;
    }

    return 0;
}