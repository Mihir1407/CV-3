// morphologicalFilter.cpp
// author: Mihir Chitre, Aditya Gurnani
// date: 02/24/2024
// description: This program captures video from a webcam, applies a morphological filter to each frame,
//              and displays the processed video in a window.

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
        return -1;
    }

    cv::namedWindow("Morphological filter", cv::WINDOW_AUTOSIZE);
    while (true) {
        cv::Mat frame, processedFrame;

        cap >> frame; 
        
        if (frame.empty())
            break;

        processedFrame = applyMorphologicalFilter(frame);

        cv::imshow("Morphological filter", processedFrame);
        
        if (cv::waitKey(30) >= 0) 
            break; 
    }

    return 0;
}
