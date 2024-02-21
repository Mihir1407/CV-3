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

cv::Mat computeFeaturesAndDraw(const cv::Mat& src) {
    // Assuming src is a cleaned binary image from applyMorphologicalFilter
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(src.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    cv::Mat output;
    cv::cvtColor(src, output, cv::COLOR_GRAY2BGR); // Convert binary to BGR for visualization

    for (size_t i = 0; i < contours.size(); i++) {
        double area = cv::contourArea(contours[i]);
        if (area < 100) // Filter out small areas
            continue;

        // Compute moments for each contour
        cv::Moments m = cv::moments(contours[i]);
        double cX = m.m10 / m.m00; // Centroid X
        double cY = m.m01 / m.m00; // Centroid Y

        // Compute the angle of the axis of least inertia
        double angle = 0.5 * atan2(2 * m.mu11, m.mu20 - m.mu02) * (180 / CV_PI);

        // Compute the oriented bounding box
        cv::RotatedRect rotatedRect = cv::minAreaRect(contours[i]);

        // Draw the centroid
        cv::circle(output, cv::Point(static_cast<int>(cX), static_cast<int>(cY)), 5, cv::Scalar(0, 255, 0), -1);

        // Draw the oriented bounding box
        cv::Point2f rectPoints[4];
        rotatedRect.points(rectPoints);
        for (int j = 0; j < 4; j++) {
            cv::line(output, rectPoints[j], rectPoints[(j+1) % 4], cv::Scalar(0, 0, 255), 2);
        }

        // Optionally display the angle or other features on the video output
        std::string text = "Angle: " + std::to_string(angle);
        cv::putText(output, text, cv::Point(static_cast<int>(cX) - 50, static_cast<int>(cY) - 20),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
    }

    return output;
}

// Main processing function adapted for feature computation and drawing
cv::Mat processFrameForFeatures(const cv::Mat& frame) {
    cv::Mat cleaned = applyMorphologicalFilter(frame);
    cv::Mat featuresImage = computeFeaturesAndDraw(cleaned);
    return featuresImage;
}

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

        // Process the frame to compute and display features for each major region
        processedFrame = processFrameForFeatures(frame);

        // Display the resulting frame with features
        cv::imshow("Features Display", processedFrame);
        if (cv::waitKey(30) >= 0) break; // Press any key to exit
    }

    return 0;
}
