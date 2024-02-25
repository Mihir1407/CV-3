#ifndef OBJECT_RECOGNITION_H
#define OBJECT_RECOGNITION_H

cv::Mat processFrameForThreshold(const cv::Mat& frame);

cv::Mat applyMorphologicalFilter(const cv::Mat& frame);

cv::Mat applyConnectedComponentsAndDisplayRegions(const cv::Mat& frame);

cv::Mat displayConnectedComponents(const cv::Mat& img, int sizeThreshold);

void findRegions(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize);

void findRegionsAndStoreToCsv(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize);

std::string classifyAndLabelRegions(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize);

std::string classifyAndLabelRegionsKNN(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize);

int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug=0);

#endif