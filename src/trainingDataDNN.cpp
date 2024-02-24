// trainingDataDNN.cpp
// Author: Mihir Chitre, Aditya Gurnani
// Date: 02/24/2024
// Description: This program captures video from a webcam, detects objects in each frame using a pre-trained ONNX model,
//              prompts the user to label the detected objects, and appends the extracted features and labels to a CSV file.

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <fstream>
#include <iostream>
#include "objectRecogFunctions.h"

/*
   Function: appendEmbeddingToCSV
   Purpose: Appends the given label and embedding to the specified CSV file.
   Arguments:
       label (const std::string&) : The label of the object.
       embedding (const cv::Mat&) : The feature embedding of the object.
       filePath (const std::string&) : The path to the CSV file.
   Returns: void
*/
void appendEmbeddingToCSV(const std::string& label, const cv::Mat& embedding, const std::string& filePath) {
    std::ofstream file(filePath, std::ios::app); 
    if (file.is_open()) {
        file << label; 
        for (int i = 0; i < embedding.total(); ++i) {
            file << "," << embedding.at<float>(i); 
        }
        file << "\n"; 
        file.close();
    } else {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
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

            //Get the embedding
            cv::Mat embedding;
            getEmbedding(grayFrame, embedding, bbox, net, 1);

            std::cout << "Enter label for captured object: ";
            std::string label;
            std::cin >> label;

            appendEmbeddingToCSV(label, embedding, csvPath);
        } else if (key >= 0) {
            break; 
        }
    }
    return 0;
}



