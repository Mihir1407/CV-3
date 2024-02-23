#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include "kmeans.h"
#include "objectRecogFunctions.h"
#include <map>
#include <fstream>
#include <iostream>

int main()
{
    cv::VideoCapture cap(0); // Open the default camera
    if (!cap.isOpened())
    { // Check if we succeeded
        std::cerr << "Error opening video capture" << std::endl;
        return -1;
    }

    cv::Mat frame, output;
    int minRegionSize = 500; // Minimum size of regions to consider
    bool trainingMode = false; // Flag to indicate if training mode is active

    while (true)
    {
        cap >> frame; // Capture a new frame
        if (frame.empty())
            break; // Check for end of video

        cv::Mat cleaned = applyMorphologicalFilter(frame); // Apply the morphological filter

        int key = cv::waitKey(30);
        if (key == 'n') // If 'n' is pressed, activate training mode for the next frame
        {
            trainingMode = true;
        }

        if (trainingMode)
        {
            findRegionsAndStoreToCsv(cleaned, output, frame, minRegionSize); // Process the frame in training mode
            trainingMode = false; // Reset training mode so that it processes only one frame
        }
        else
        {
            findRegions(cleaned, output, frame, minRegionSize); // Normal processing
        }

        cv::imshow("Output", output); // Display the output image with computed features
        if (key >= 0 && key != 'n')
            break; // Press any key except 'n' to exit
    }
    return 0;
}