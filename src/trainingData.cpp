// trainingData.cpp
// Author: Mihir Chitre, Aditya Gurnani
// Date: 02/24/2024
// Description: This program captures video from a webcam, processes each frame to identify regions, computes the feature vectors for biggest central region
//              and optionally enters a training mode to store feature vectors to a CSV file.


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
    bool trainingMode = false; // Flag to indicate if training mode is active

    while (true)
    {
        cap >> frame; 
        if (frame.empty())
            break; 

        cv::Mat cleaned = applyMorphologicalFilter(frame);

        int key = cv::waitKey(30);
        if (key == 'n') 
        {
            trainingMode = true;
        }

        if (trainingMode)
        {
            findRegionsAndStoreToCsv(cleaned, output, frame, minRegionSize); 
            trainingMode = false; 
        }
        else
        {
            findRegions(cleaned, output, frame, minRegionSize); 
        }

        cv::imshow("Output", output); 
        if (key >= 0 && key != 'n')
            break; 
    }
    return 0;
}