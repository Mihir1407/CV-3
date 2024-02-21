#include <opencv2/opencv.hpp>
#include <iostream>

void customThreshold(const cv::Mat& input, cv::Mat& output, int thresholdValue) {
    cv::cvtColor(input, output, cv::COLOR_BGR2GRAY); // Convert to grayscale
    for (int i = 0; i < output.rows; ++i) {
        for (int j = 0; j < output.cols; ++j) {
            output.at<uchar>(i, j) = (output.at<uchar>(i, j) > thresholdValue) ? 255 : 0;
        }
    }
}

int main() {
    // Setup VideoCapture to use your iPhone camera as source
    cv::VideoCapture cap(1);; // You might need to change this ID or use an IP stream address

    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat processedFrame;
        customThreshold(frame, processedFrame, 127); // Example threshold value

        // Display the processed frame
        cv::imshow("Processed Frame", processedFrame);

        if (cv::waitKey(30) >= 0) break; // Break the loop on any key press
    }

    return 0;
}
