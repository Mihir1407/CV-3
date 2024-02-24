// classifyImagesDNN.cpp
// Author: Mihir Chitre, Aditya Gurnani
// Date: 02/24/2024
// Description: This program captures video from a webcam, detects objects in each frame using a pre-trained ONNX model, computes embeddings for it,
//              reads the csv database of computed embeddings, it then compares the computed embedding with embeddings in database using cosine distance,
//              and assigns the label of best matching embedding to the object in the frame.
//               

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "objectRecogFunctions.h"

/*
   Function: cosineDistance
   Purpose: Calculate the cosine distance between two vectors.
   Arguments:
       vec1 (const cv::Mat&) : First vector.
       vec2 (const cv::Mat&) : Second vector.
   Returns:
       double : Cosine distance between the two vectors.
*/
double cosineDistance(const cv::Mat& vec1, const cv::Mat& vec2) {
    double dot = vec1.dot(vec2);
    double denom = cv::norm(vec1) * cv::norm(vec2);
    return 1.0 - (dot / denom); 
}

/*
   Function: loadEmbeddingsAndLabels
   Purpose: Load embeddings and their corresponding labels from a CSV file.
   Arguments:
       filePath (const std::string&) : Path to the CSV file.
   Returns:
       std::vector<std::pair<cv::Mat, std::string>> : Vector of pairs containing embeddings and labels.
*/
std::vector<std::pair<cv::Mat, std::string>> loadEmbeddingsAndLabels(const std::string& filePath) {
    std::vector<std::pair<cv::Mat, std::string>> database;
    std::ifstream file(filePath);
    std::string line, label;
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::getline(iss, label, ',');
        std::vector<float> embeddingValues;
        std::string value;
        while (std::getline(iss, value, ',')) {
            embeddingValues.push_back(std::stof(value));
        }
        cv::Mat embedding(1, embeddingValues.size(), CV_32F, embeddingValues.data());
        database.push_back({embedding.clone(), label});
    }
    return database;
}

/*
   Function: main
   Purpose: Entry point of the program.
   Returns: 0 on successful execution, -1 otherwise.
*/
int main() {
    std::string modelPath = "or2d-normmodel-007.onnx";
    std::string csvPath = "feature_vectors_DNN.csv"; 
    cv::dnn::Net net = cv::dnn::readNet(modelPath);

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video capture" << std::endl;
        return -1;
    }

    auto database = loadEmbeddingsAndLabels(csvPath);

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::imshow("Frame", frame);
        int key = cv::waitKey(1);
        if (key == 'n') {

            cv::Mat grayFrame;
            grayFrame = applyMorphologicalFilter(frame);

            cv::Rect bbox(0, 0, grayFrame.cols, grayFrame.rows);
            cv::Mat embedding;
            getEmbedding(grayFrame, embedding, bbox, net, 0); 

            // Find the closest matching label using cosine distance
            std::string closestLabel = "Unknown";
            double closestDistance = std::numeric_limits<double>::max();
            for (const auto& entry : database) {
                double distance = cosineDistance(embedding, entry.first);
                if (distance < closestDistance) {
                    closestDistance = distance;
                    closestLabel = entry.second;
                }
            }

            std::cout << "Closest label: " << closestLabel << std::endl;
            cv::putText(frame, closestLabel, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
            cv::imshow("Frame", frame);
            cv::waitKey(0); 
        } else if (key >= 0) {
            break; 
        }
    }
    return 0;
}