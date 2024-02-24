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

// int main()
// {
//     cv::Mat frame, output;
//     int minRegionSize = 500;
//     frame = cv::imread("D:/CV/Project3/Proj03Examples/img3P3.png");
//     if (frame.empty())
//     {
//         std::cout << "Could not open or find the image" << std::endl;
//         return -1;
//     }
//     cv::Mat cleaned = applyMorphologicalFilter(frame); // Apply the morphological filter

//     findRegions(cleaned, output, frame, minRegionSize); // Find and analyze regions

//     cv::imshow("Display window", output);

//     while (true)
//     {
//         char key = cv::waitKey(0);
//         if (key == 'q')
//         {
//             break;
//         }
//     }
//     return 0;
// }

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

    while (true)
    {
        cap >> frame; // Capture a new frame
        if (frame.empty())
            break; // Check for end of video

        cv::Mat cleaned = applyMorphologicalFilter(frame); // Apply the morphological filter

        classifyAndLabelRegions(cleaned, output, frame, minRegionSize); // Find and analyze regions

        cv::imshow("Output", frame); // Display the output image with computed features
        if (cv::waitKey(30) >= 0)
            break; // Press any key to exit
    }
    return 0;
}