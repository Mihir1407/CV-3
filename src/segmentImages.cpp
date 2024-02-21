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
    cv::VideoCapture cap(0); // Change to the correct video source if needed
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    while(true) {
        cv::Mat frame, processedFrame;
        cap >> frame; // Capture frame-by-frame
        if (frame.empty()) break;

        // Process the frame to apply morphological filtering and display region maps
        processedFrame = applyConnectedComponentsAndDisplayRegions(frame);

        // Display the resulting frame
        cv::imshow("Region Maps", processedFrame);
        if (cv::waitKey(30) >= 0) break; // Press any key to exit
    }

    return 0;
}
