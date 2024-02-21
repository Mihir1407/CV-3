#ifndef OBJECT_RECOGNITION_H
#define OBJECT_RECOGNITION_H

cv::Mat processFrameForThreshold(const cv::Mat& frame);

cv::Mat applyMorphologicalFilter(const cv::Mat& frame);

cv::Mat applyConnectedComponentsAndDisplayRegions(const cv::Mat& frame);

#endif