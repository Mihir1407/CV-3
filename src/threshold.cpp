// threshold.cpp
// author: Mihir Chitre, Aditya Gurnani
// date: 02/24/2024
// description: This program captures video from a webcam, processes each frame to obtain a thresholded version,
//              and displays the thresholded video in a window.

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

    cv::namedWindow("Thresholded Video", cv::WINDOW_AUTOSIZE);
    while (true) {
        cv::Mat frame, processedFrame;
        cap >> frame; 
        
        if (frame.empty())
            break;

        processedFrame = processFrameForThreshold(frame);

        cv::imshow("Thresholded Video", processedFrame);
        if (cv::waitKey(30) >= 0) break; 
    }

    return 0;
}
