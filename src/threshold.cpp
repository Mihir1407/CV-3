#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include "kmeans.h"
#include "objectRecogFunctions.h"

int main() {
    cv::VideoCapture cap(0); 
    if (!cap.isOpened()) {
        return -1;
    }

    cv::namedWindow("Thresholded Video", cv::WINDOW_AUTOSIZE);
    while (true) {
        cv::Mat frame, processedFrame;
        cap >> frame; // Get a new frame from camera
        
        if (frame.empty())
            break;

        // Process the frame
        processedFrame = processFrameForThreshold(frame);

        // Show the result
        cv::imshow("Thresholded Video", processedFrame);
        if (cv::waitKey(30) >= 0) break; // Wait for a key press
    }

    return 0;
}
