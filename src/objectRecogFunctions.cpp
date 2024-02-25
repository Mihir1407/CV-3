// objectRecogFunctions.cpp
// Author: Mihir Chitre, Aditya Gurnani
// Date: 02/24/2024
// Description: This program is designed to process images for object recognition tasks. 
//              It includes utilities for grayscale conversion, Gaussian blurring, K-means thresholding, and morphological operations.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include "kmeans.h"
#include "objectRecogFunctions.h"
#include <corecrt_math_defines.h>

/*
   Function: convertToGrayscale
   Purpose: Converts a color image to grayscale using a weighted sum method. The weights for the red, green, and blue channels are based on their perceived luminance.
   Arguments:
       src (const cv::Mat&) : The source color image.
   Returns: cv::Mat : The converted grayscale image.
*/
cv::Mat convertToGrayscale(const cv::Mat &src)
{
    CV_Assert(src.type() == CV_8UC3);
    cv::Mat grayscale(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            uchar gray = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            grayscale.at<uchar>(i, j) = gray;
        }
    }
    return grayscale;
}

/*
   Function: applyGaussianBlur
   Purpose: Applies Gaussian Blur to the input image, smoothing it to reduce noise and details.
   Arguments: 
       src (const cv::Mat&) : The source image to be blurred.
       kernelSize (int) : The size of the Gaussian kernel.
       sigma (double) : The standard deviation of the Gaussian kernel in both X and Y direction.
   Returns: cv::Mat : The blurred image.
*/
cv::Mat applyGaussianBlur(const cv::Mat &src, int kernelSize, double sigma)
{
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(kernelSize, kernelSize), sigma, sigma);
    return blurred; 
    }

/*
   Function: kMeansThreshold
   Purpose: Implements a simple K-means algorithm to find an optimal threshold for binarizing an image. 
   Arguments:
       src (const cv::Mat&) : The source grayscale image.
       maxIterations (int) : The maximum number of iterations for the K-means algorithm.
   Returns: int : The calculated threshold value.
*/
int kMeansThreshold(const cv::Mat &src, int maxIterations = 10)
{
    std::vector<int> samples;
    for (int i = 0; i < src.rows; i += 4)
    {
        for (int j = 0; j < src.cols; j += 4)
        {
            samples.push_back(src.at<uchar>(i, j));
        }
    }

    int centroid1 = 255, centroid2 = 0;
    for (int it = 0; it < maxIterations; ++it)
    {
        std::vector<int> cluster1, cluster2;
        for (int val : samples)
        {
            if (std::abs(val - centroid1) < std::abs(val - centroid2))
            {
                cluster1.push_back(val);
            }
            else
            {
                cluster2.push_back(val);
            }
        }
        centroid1 = cluster1.empty() ? centroid1 : std::accumulate(cluster1.begin(), cluster1.end(), 0) / cluster1.size();
        centroid2 = cluster2.empty() ? centroid2 : std::accumulate(cluster2.begin(), cluster2.end(), 0) / cluster2.size();
    }
    return (centroid1 + centroid2) / 2; 
}

/*
   Function: applyThreshold
   Purpose: Applies binary thresholding to an image based on a specified threshold value. 
   Arguments:
       src (const cv::Mat&) : The source image to be thresholded.
       threshold (int) : The threshold value.
   Returns: cv::Mat : The thresholded binary image.
*/
cv::Mat applyThreshold(const cv::Mat &src, int threshold)
{
    cv::Mat thresholded(src.size(), src.type());
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            thresholded.at<uchar>(i, j) = (src.at<uchar>(i, j) > threshold) ? 255 : 0;
        }
    }
    return thresholded;
}

/*
   Function: processFrameForThreshold
   Purpose: The purpose includes converting the frame to grayscale, applying Gaussian blur, determining an optimal threshold using K-means, and applying the threshold.
   Arguments:
       frame (const cv::Mat&) : The input frame to be processed.
   Returns: cv::Mat : The processed frame ready for further analysis.
*/
cv::Mat processFrameForThreshold(const cv::Mat &frame)
{
    cv::Mat grayscale = convertToGrayscale(frame);
    cv::Mat blurred = applyGaussianBlur(grayscale, 5, 1.5); // Kernel size and sigma for Gaussian blur
    int thresholdValue = kMeansThreshold(blurred);
    cv::Mat thresholded = applyThreshold(blurred, thresholdValue);
    return thresholded;
}

/*
   Function: applyErosion
   Purpose: Applies morphological erosion to the input image, removes small white noises, detaches small islands of pixels, and shrink objects.
   Arguments:
       src (const cv::Mat&) : Input image on which erosion is applied.
       kernelSize (int) : Size of the kernel used for erosion.
   Returns: cv::Mat : The eroded output image.
*/
cv::Mat applyErosion(const cv::Mat &src, int kernelSize)
{
    cv::Mat eroded(src.size(), src.type(), cv::Scalar::all(0));
    int k = kernelSize / 2; 
    for (int i = k; i < src.rows - k; ++i)
    {
        for (int j = k; j < src.cols - k; ++j)
        {
            uchar min = 255;
            for (int ki = -k; ki <= k; ++ki)
            {
                for (int kj = -k; kj <= k; ++kj)
                {
                    uchar val = src.at<uchar>(i + ki, j + kj);
                    if (val < min)
                        min = val;
                }
            }
            eroded.at<uchar>(i, j) = min == 255 ? 255 : 0;
        }
    }
    return eroded;
}

/*
   Function: applyDilation
   Purpose: Applies morphological dilation to the input image, which helps in accentuating features and joining broken parts of objects. 
   Arguments:
       src (const cv::Mat&) : Input image on which dilation is applied.
       kernelSize (int) : Size of the kernel used for dilation.
   Returns: cv::Mat : The dilated output image.
*/
cv::Mat applyDilation(const cv::Mat &src, int kernelSize)
{
    cv::Mat dilated(src.size(), src.type(), cv::Scalar::all(0));
    int k = kernelSize / 2; 
    for (int i = k; i < src.rows - k; ++i)
    {
        for (int j = k; j < src.cols - k; ++j)
        {
            uchar max = 0;
            for (int ki = -k; ki <= k; ++ki)
            {
                for (int kj = -k; kj <= k; ++kj)
                {
                    uchar val = src.at<uchar>(i + ki, j + kj);
                    if (val > max)
                        max = val;
                }
            }
            dilated.at<uchar>(i, j) = max == 0 ? 0 : 255;
        }
    }
    return dilated;
}

/*
   Function: cleanupBinaryImage
   Purpose: Cleans up a binary image by applying erosion followed by dilation, known as opening. 
   Arguments:
       src (const cv::Mat&) : Input binary image to clean up.
       erosionSize (int) : Size of the kernel used for erosion.
       dilationSize (int) : Size of the kernel used for dilation.
   Returns: cv::Mat : The cleaned-up output image.
*/
cv::Mat cleanupBinaryImage(const cv::Mat &src, int erosionSize = 3, int dilationSize = 3)
{
    cv::Mat eroded = applyErosion(src, erosionSize);
    cv::Mat cleaned = applyDilation(eroded, dilationSize);
    return cleaned;
}

/*
   Function: applyMorphologicalFilter
   Purpose: Applies a morphological filter to the frame using processFrameForThreshold function, then cleaning up using cleanupBinaryImage function. 
   Arguments:
       frame (const cv::Mat&) : The frame to be processed.
   Returns: cv::Mat : The output image after applying the morphological filter.
*/
cv::Mat applyMorphologicalFilter(const cv::Mat &frame)
{
    cv::Mat thresholded = processFrameForThreshold(frame);
    cv::Mat cleaned = cleanupBinaryImage(thresholded, 7, 7);
    return cleaned;
}

/*
   Function: UnionFind
   Purpose: Implements the Union-Find algorithm, also known as the Disjoint Set Union algorithm, to track a set of elements partitioned into a number of disjoint subsets. 
   Members:
       parent (std::vector<int>) : Vector holding the parent of each node.
   Methods:
       find(int) : Finds the representative of the set that an element belongs to.
       unite(int, int) : Merges two subsets into a single subset.
*/
struct UnionFind
{
    std::vector<int> parent;

    UnionFind(int n) : parent(n)
    {
        for (int i = 0; i < n; ++i)
            parent[i] = i;
    }

    int find(int x)
    {
        if (parent[x] == x)
            return x;
        return parent[x] = find(parent[x]);
    }

    void unite(int x, int y)
    {
        x = find(x);
        y = find(y);
        if (x != y)
            parent[x] = y;
    }
};

/*
   Function: findMinNeighborLabel
   Purpose: Finds the minimum label among the neighbors of a given pixel in the labeled image. 
   Arguments:
       labels (const cv::Mat&) : The matrix containing labels of pixels.
       i (int) : Row index of the current pixel.
       j (int) : Column index of the current pixel.
   Returns: int : The minimum label found among the neighbors.
*/
int findMinNeighborLabel(const cv::Mat &labels, int i, int j)
{
    std::vector<int> neighbors;
    // Check top and left (add top-right and left-bottom if 8-connectivity)
    if (i > 0 && labels.at<int>(i - 1, j) > 0)
        neighbors.push_back(labels.at<int>(i - 1, j));
    if (j > 0 && labels.at<int>(i, j - 1) > 0)
        neighbors.push_back(labels.at<int>(i, j - 1));
    if (neighbors.empty())
        return 0; // No neighbors
    return *min_element(neighbors.begin(), neighbors.end());
}

/*
   Function: applyConnectedComponents
   Purpose: Labels connected components in a binary image. Pixels with the same label are connected and pixels with different labels are not connected.
   Arguments:
       src (const cv::Mat&) : Input binary image.
       minSize (int) : Minimum size of components to retain.
   Returns: cv::Mat : The output image with labeled connected components.
*/
cv::Mat applyConnectedComponents(const cv::Mat &src, int minSize)
{
    cv::Mat labels = cv::Mat::zeros(src.size(), CV_32S);
    int nextLabel = 1;
    UnionFind uf(10000); // Arbitrary large size

    // First Pass
    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            if (src.at<uchar>(i, j) == 255)
            { 
                int label = findMinNeighborLabel(labels, i, j);
                if (label == 0)
                {
                    label = nextLabel++;
                }
                else
                {
                    if (i > 0 && labels.at<int>(i - 1, j) > 0)
                        uf.unite(label, labels.at<int>(i - 1, j));
                    if (j > 0 && labels.at<int>(i, j - 1) > 0)
                        uf.unite(label, labels.at<int>(i, j - 1));
                }
                labels.at<int>(i, j) = label;
            }
        }
    }

    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            int label = labels.at<int>(i, j);
            if (label > 0)
                labels.at<int>(i, j) = uf.find(label);
        }
    }

    std::map<int, int> labelSizes;
    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            int label = labels.at<int>(i, j);
            if (label > 0)
                labelSizes[label]++;
        }
    }

    std::vector<cv::Vec3b> colors(nextLabel + 1);
    std::generate(colors.begin(), colors.end(), []()
                  { return cv::Vec3b(rand() % 256, rand() % 256, rand() % 256); });

    cv::Mat output = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            int label = labels.at<int>(i, j);
            if (label > 0 && labelSizes[label] >= minSize)
            {
                output.at<cv::Vec3b>(i, j) = colors[label];
            }
        }
    }

    return output;
}

/*
   Function: applyConnectedComponentsAndDisplayRegions
   Purpose: Applies connected components labeling to a binary image and filters out small regions.
   Arguments:
       frame (const cv::Mat&) : The frame to be processed.
   Returns: cv::Mat : The output image with labeled and filtered connected components.
*/
cv::Mat applyConnectedComponentsAndDisplayRegions(const cv::Mat &frame)
{
    cv::Mat cleaned = applyMorphologicalFilter(frame); 
    cv::Mat labeledRegions = applyConnectedComponents(cleaned, 50); 

    return labeledRegions;
}

/*
   Function: displayConnectedComponents
   Purpose: Visualizes the connected components of a binary image by assigning a unique color to each component. 
   Arguments:
       img (const cv::Mat&) : Input binary image.
       sizeThreshold (int) : Minimum size of components to display.
   Returns: cv::Mat : The output image with colored connected components.
*/
cv::Mat displayConnectedComponents(const cv::Mat &img, int sizeThreshold = 100)
{
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(img, labels, stats, centroids, 8, CV_32S);

    static std::map<int, cv::Vec3b> labelColors; 
    labelColors.clear();                         

    for (int label = 1; label < nLabels; ++label)
    {
        int area = stats.at<int>(label, cv::CC_STAT_AREA);
        if (area >= sizeThreshold)
        {
            // Ensure each label has a unique color
            if (labelColors.find(label) == labelColors.end())
            {
                labelColors[label] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
            }
        }
    }

    cv::Mat dst = cv::Mat::zeros(img.size(), CV_8UC3); // Output image

    for (int r = 0; r < labels.rows; ++r)
    {
        for (int c = 0; c < labels.cols; ++c)
        {
            int label = labels.at<int>(r, c);
            if (labelColors.find(label) != labelColors.end())
            {
                dst.at<cv::Vec3b>(r, c) = labelColors[label];
            }
        }
    }

    return dst;
}

/*
   Function: drawOrientedBoundingBox
   Purpose: Draws an oriented bounding box around the white pixels in a binary mask. 
   Arguments:
       mask (const cv::Mat&) : Input binary mask where the object is white.
       output (cv::Mat&) : The image on which the oriented bounding box will be drawn.
   Returns: void
*/
void drawOrientedBoundingBox(const cv::Mat &mask, cv::Mat &output)
{
    std::vector<cv::Point> points;
    for (int y = 0; y < mask.rows; y++)
    {
        for (int x = 0; x < mask.cols; x++)
        {
            if (mask.at<uchar>(y, x) == 255)
            { 
                points.push_back(cv::Point(x, y));
            }
        }
    }

    cv::RotatedRect rotatedRect = cv::minAreaRect(points);
    cv::Point2f vertices[4];
    rotatedRect.points(vertices);
    for (int i = 0; i < 4; i++)
        cv::line(output, vertices[i], vertices[(i + 1) % 4], cv::Scalar(0, 0, 255), 2);
}

/*
   Function: drawAxisOfLeastCentralMoment
   Purpose: Draws the axis of least inertia for an object in a binary mask, for understanding the object's orientation.
   Arguments:
       mask (const cv::Mat&) : Input binary mask where the object is white.
       output (cv::Mat&) : The image on which the axis will be drawn.
   Returns: void
*/
void drawAxisOfLeastCentralMoment(const cv::Mat &mask, cv::Mat &output)
{
    std::vector<cv::Point> points;
    for (int y = 0; y < mask.rows; y++)
    {
        for (int x = 0; x < mask.cols; x++)
        {
            if (mask.at<uchar>(y, x) == 255)
            { 
                points.push_back(cv::Point(x, y));
            }
        }
    }

    if (points.empty())
        return; 

    cv::Mat data = cv::Mat(points.size(), 2, CV_32F);
    for (size_t i = 0; i < points.size(); ++i)
    {
        data.at<float>(i, 0) = points[i].x;
        data.at<float>(i, 1) = points[i].y;
    }

    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);
    cv::Point2f center(pca.mean.at<float>(0), pca.mean.at<float>(1));
    cv::Vec2f eigenvector = pca.eigenvectors.at<float>(0); 
    cv::Point endpoint = center + cv::Point2f(eigenvector[0] * 100, eigenvector[1] * 100);
    cv::line(output, center, endpoint, cv::Scalar(255, 0, 0), 2);
}

/*
   Function: computeFeatureVector
   Purpose: Computes a feature vector for an object in a binary mask. 
   Arguments:
       mask (const cv::Mat&) : Input binary mask of the object.
       area (const int) : The area of the object.
       width (const int) : The width of the bounding box of the object.
       height (const int) : The height of the bounding box of the object.
       labels (const cv::Mat&) : The labeled image containing the object.
       objectLabel (const int) : The label of the object in the labeled image.
   Returns: std::unordered_map<std::string, double> : The computed feature vector.
*/
std::unordered_map<std::string, double> computeFeatureVector(const cv::Mat &mask, const int area, const int width, const int height, const cv::Mat &labels, const int objectLabel)
{
    std::unordered_map<std::string, double> featureMap;

    double percentFilled = ((double)area / (width * height) * 100);
    double aspectRatio = (double)height / width;

    featureMap["area"] = area;
    featureMap["percentFilled"] = percentFilled;
    featureMap["aspectRatio"] = aspectRatio;

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double perimeter = 0;
    if (!contours.empty())
    {
        perimeter = cv::arcLength(contours[0], true); 
    }

    double circularity = (4 * M_PI * area) / (perimeter * perimeter);
    double compactness = sqrt((4 * area) / M_PI) / std::max(width, height);

    featureMap["circularity"] = circularity;
    featureMap["compactness"] = compactness;

    cv::Moments objMoments = cv::moments(mask, true);
    double huMoments[7];
    cv::HuMoments(objMoments, huMoments);

    for (int i = 0; i < 7; i++)
    {
        featureMap["HuMoment " + std::to_string(i)] = -1 * copysign(1.0, huMoments[i]) * log10(std::abs(huMoments[i]));
    }

    return featureMap;
}

/*
   Function: drawFeatures
   Purpose: Draws the calculated features on the output image for visualization.
   Arguments:
       output (cv::Mat&) : The image on which features will be drawn.
       featureMap (const std::unordered_map<std::string, double>&) : The feature vector of the object.
       mask (const cv::Mat&) : The binary mask of the object.
       x (const int) : The x-coordinate of the top-left corner of the object's bounding box.
       y (const int) : The y-coordinate of the top-left corner of the object's bounding box.
   Returns: void
*/
void drawFeatures(cv::Mat &output, const std::unordered_map<std::string, double> &featureMap, const cv::Mat &mask, const int x, const int y)
{
    cv::putText(output, cv::format("Circularity: %.2f, Compactness: %.2f", featureMap.at("circularity"), featureMap.at("compactness")),
                cv::Point(x, y - 60), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    cv::putText(output, cv::format("Area: %d, Bounding Box Ratio: %.2f", int(featureMap.at("area")), featureMap.at("aspectRatio")),
                cv::Point(x, y - 40), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    cv::putText(output, cv::format("Percentage Filled: %.2f", featureMap.at("percentFilled")),
                cv::Point(x, y - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    drawOrientedBoundingBox(mask, output);
    drawAxisOfLeastCentralMoment(mask, output);
}

/*
   Function: findRegions
   Purpose: Identifies and analyzes regions in a binary image. 
   Arguments:
       binaryImage (const cv::Mat&) : Input binary image.
       output (cv::Mat&) : The image where regions will be visualized.
       originalImg (cv::Mat&) : The original image for reference.
       minRegionSize (int) : The minimum size of regions to consider.
   Returns: void
*/
void findRegions(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize)
{
    cv::Mat invertedImg;
    cv::bitwise_not(binaryImage, invertedImg);
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(invertedImg, labels, stats, centroids, 8, CV_32S);

    output = cv::Mat::zeros(originalImg.size(), CV_8UC3);

    cv::Point imageCenter = cv::Point(originalImg.cols / 2, originalImg.rows / 2);

    int centerAreaSize = std::min(originalImg.cols, originalImg.rows) / 3;
    cv::Rect centerArea(imageCenter.x - centerAreaSize, imageCenter.y - centerAreaSize, centerAreaSize * 2, centerAreaSize * 2);
    cv::rectangle(originalImg, centerArea, cv::Scalar(255, 255, 0), 2);

    std::vector<int> objectsLabel;
    double minDistanceToCenter = std::numeric_limits<double>::max();

    for (int i = 1; i < nLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area > minRegionSize)
        {
            cv::Point centroid = cv::Point(static_cast<int>(centroids.at<double>(i, 0)),
                                           static_cast<int>(centroids.at<double>(i, 1)));

            double distance = cv::norm(centroid - imageCenter);

            if (centerArea.contains(centroid))
            {
                objectsLabel.push_back(i);
            }
        }
    }

    std::vector<cv::Vec3b> colors(objectsLabel.size() + 1);

    for (size_t i = 0; i < objectsLabel.size(); i++)
    {
        colors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
    }

    for (size_t c = 0; c < objectsLabel.size(); c++)
    {
        int area = stats.at<int>(objectsLabel[c], cv::CC_STAT_AREA);
        int x = stats.at<int>(objectsLabel[c], cv::CC_STAT_LEFT);
        int y = stats.at<int>(objectsLabel[c], cv::CC_STAT_TOP);
        int width = stats.at<int>(objectsLabel[c], cv::CC_STAT_WIDTH);
        int height = stats.at<int>(objectsLabel[c], cv::CC_STAT_HEIGHT);

        cv::Mat mask = cv::Mat::zeros(originalImg.size(), CV_8UC1);
        for (int i = 0; i < mask.rows; i++)
        {
            uchar *maskptr = mask.ptr<uchar>(i);
            for (int j = 0; j < mask.cols; j++)
            {
                if (labels.at<int>(i, j) == objectsLabel[c])
                {
                    maskptr[j] = 255;
                }
            }
        }

        auto featureMap = computeFeatureVector(mask, area, width, height, labels, objectsLabel[c]);
        drawFeatures(output, featureMap, mask, x, y);
        for (int i = 0; i < originalImg.rows; i++)
        {
            for (int j = 0; j < originalImg.cols; j++)
            {
                if (labels.at<int>(i, j) == objectsLabel[c])
                {
                    output.at<cv::Vec3b>(i, j) = colors[c];
                }
            }
        }
    }
}

/*
   Function: saveFeatureVectorToCSV
   Purpose: Saves the computed feature vector of an object along with its label to a CSV file. 
   Arguments:
       featureMap (const std::unordered_map<std::string, double>&) : The feature vector to save.
       label (const std::string&) : The label of the object.
   Returns: void
*/
void saveFeatureVectorToCSV(const std::unordered_map<std::string, double> &featureMap, const std::string &label)
{
    std::ofstream file("feature_vectors.csv", std::ios::app); 
    if (file.is_open())
    {
        std::vector<std::string> orderedFeatureKeys = {
            "area", "percentFilled", "aspectRatio", "circularity", "compactness",
            "HuMoment 0", "HuMoment 1", "HuMoment 2", "HuMoment 3",
            "HuMoment 4", "HuMoment 5", "HuMoment 6"};

        file << label; 

        for (const auto &key : orderedFeatureKeys)
        {
            if (featureMap.find(key) != featureMap.end())
            {                                      
                file << "," << featureMap.at(key); 
            }
            else
            {
                file << ","; 
            }
        }

        file << "\n"; 
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

/*
   Function: findRegionsAndStoreToCsv
   Purpose: Identifies regions in a binary image, computes feature vectors, prompts for labels, and stores the feature vectors along with labels to a CSV file. 
   Arguments:
       binaryImage (const cv::Mat&) : Input binary image.
       output (cv::Mat&) : The image where regions will be visualized.
       originalImg (cv::Mat&) : The original image for reference.
       minRegionSize (int) : The minimum size of regions to consider.
   Returns: void
*/
void findRegionsAndStoreToCsv(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize)
{
    cv::Mat invertedImg;
    cv::bitwise_not(binaryImage, invertedImg);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(invertedImg, labels, stats, centroids, 8, CV_32S);

    output = cv::Mat::zeros(originalImg.size(), CV_8UC3);

    cv::Point imageCenter = cv::Point(originalImg.cols / 2, originalImg.rows / 2);

    int centerAreaSize = std::min(originalImg.cols, originalImg.rows) / 3;
    cv::Rect centerArea(imageCenter.x - centerAreaSize, imageCenter.y - centerAreaSize, centerAreaSize * 2, centerAreaSize * 2);
    cv::rectangle(originalImg, centerArea, cv::Scalar(255, 255, 0), 2);

    std::vector<int> objectsLabel;
    double minDistanceToCenter = std::numeric_limits<double>::max();

    for (int i = 1; i < nLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        if (area > minRegionSize)
        {
            cv::Point centroid = cv::Point(static_cast<int>(centroids.at<double>(i, 0)),
                                           static_cast<int>(centroids.at<double>(i, 1)));

            double distance = cv::norm(centroid - imageCenter);

            if (centerArea.contains(centroid))
            {
                objectsLabel.push_back(i);
            }
        }
    }

    std::vector<cv::Vec3b> colors(objectsLabel.size() + 1);

    for (size_t i = 0; i < objectsLabel.size(); i++)
    {
        colors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
    }

    for (size_t c = 0; c < objectsLabel.size(); c++)
    {
        int area = stats.at<int>(objectsLabel[c], cv::CC_STAT_AREA);
        int x = stats.at<int>(objectsLabel[c], cv::CC_STAT_LEFT);
        int y = stats.at<int>(objectsLabel[c], cv::CC_STAT_TOP);
        int width = stats.at<int>(objectsLabel[c], cv::CC_STAT_WIDTH);
        int height = stats.at<int>(objectsLabel[c], cv::CC_STAT_HEIGHT);

        cv::Mat mask = cv::Mat::zeros(originalImg.size(), CV_8UC1);
        for (int i = 0; i < mask.rows; i++)
        {
            uchar *maskptr = mask.ptr<uchar>(i);
            for (int j = 0; j < mask.cols; j++)
            {
                if (labels.at<int>(i, j) == objectsLabel[c])
                {
                    maskptr[j] = 255;
                }
            }
        }

        auto featureMap = computeFeatureVector(mask, area, width, height, labels, objectsLabel[c]);
        std::string label;
        std::cout << "Enter label for the detected object: ";
        std::cin >> label;
        saveFeatureVectorToCSV(featureMap, label);
        drawFeatures(output, featureMap, mask, x, y);
        for (int i = 0; i < originalImg.rows; i++)
        {
            for (int j = 0; j < originalImg.cols; j++)
            {
                if (labels.at<int>(i, j) == objectsLabel[c])
                {
                    output.at<cv::Vec3b>(i, j) = colors[c];
                }
            }
        }
    }
}

/*
   Function: loadFeatureVectorsAndLabels
   Purpose: Loads feature vectors and their corresponding labels from a CSV file. 
   Arguments:
       fileName (const std::string&) : The name of the CSV file to load from.
   Returns: std::vector<std::pair<std::vector<double>, std::string>> : The loaded dataset.
*/
std::vector<std::pair<std::vector<double>, std::string>> loadFeatureVectorsAndLabels(const std::string &fileName)
{
    std::vector<std::pair<std::vector<double>, std::string>> database;
    std::ifstream file(fileName); 
    std::string line;

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return database;
    }

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string label;
        std::getline(iss, label, ','); 

        std::vector<double> features;
        std::string value;
        while (std::getline(iss, value, ','))
        {
            features.push_back(std::stod(value));
        }

        database.push_back({features, label});
    }

    return database;
}

/*
   Function: convertFeatureMapToVector
   Purpose: Converts a feature map (unordered_map) to a vector in a specified order. 
   Arguments:
       featureMap (const std::unordered_map<std::string, double>&) : The feature map to convert.
       orderedFeatureKeys (const std::vector<std::string>&) : The order in which features should appear in the vector.
   Returns: std::vector<double> : The ordered feature vector.
*/
std::vector<double> convertFeatureMapToVector(const std::unordered_map<std::string, double> &featureMap, const std::vector<std::string> &orderedFeatureKeys)
{
    std::vector<double> featureVector;
    for (const auto &key : orderedFeatureKeys)
    {
        if (featureMap.find(key) != featureMap.end())
        {
            featureVector.push_back(featureMap.at(key));
        }
        else
        {
            featureVector.push_back(0.0); 
        }
    }
    return featureVector;
}

/*
   Function: calculateStandardDeviations
   Purpose: Calculates the standard deviations of features across a dataset. 
   Arguments:
       database (const std::vector<std::pair<std::vector<double>, std::string>>&) : The dataset to analyze.
   Returns: std::vector<double> : The standard deviations of each feature across the dataset.
*/
std::vector<double> calculateStandardDeviations(const std::vector<std::pair<std::vector<double>, std::string>> &database)
{
    if (database.empty())
        return {};

    size_t numFeatures = database[0].first.size();
    std::vector<double> means(numFeatures, 0.0);
    std::vector<double> stdevs(numFeatures, 0.0);

    for (const auto &entry : database)
    {
        for (size_t i = 0; i < numFeatures; ++i)
        {
            means[i] += entry.first[i];
        }
    }
    for (double &mean : means)
        mean /= database.size();

    for (const auto &entry : database)
    {
        for (size_t i = 0; i < numFeatures; ++i)
        {
            stdevs[i] += std::pow(entry.first[i] - means[i], 2);
        }
    }
    for (double &stdev : stdevs)
        stdev = std::sqrt(stdev / database.size());

    return stdevs;
}

/*
   Function: scaledEuclideanDistance
   Purpose: Calculates the scaled Euclidean distance between two feature vectors, taking into account the standard deviation of each feature. 
   Arguments:
       vec1 (const std::vector<double>&) : The first feature vector.
       vec2 (const std::vector<double>&) : The second feature vector.
       stdevs (const std::vector<double>&) : The standard deviations of each feature, used for scaling.
   Returns: double : The scaled Euclidean distance between the two vectors.
*/
double scaledEuclideanDistance(const std::vector<double> &vec1, const std::vector<double> &vec2, const std::vector<double> &stdevs)
{
    double distance = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        if (stdevs[i] > 0)
        {
            double scaledDiff = (vec1[i] - vec2[i]) / stdevs[i];
            distance += scaledDiff * scaledDiff;
        }
    }
    return std::sqrt(distance);
}

/*
   Function: classifyAndLabelRegions
   Purpose: Classifies and labels regions in a binary image based on a dataset of feature vectors and labels. 
   Arguments:
       binaryImage (const cv::Mat&) : Input binary image.
       output (cv::Mat&) : The image where labeled regions will be visualized.
       originalImg (cv::Mat&) : The original image for reference.
       minRegionSize (int) : The minimum size of regions to consider for classification.
   Returns: std::string : The label of the classified region.
*/
std::string classifyAndLabelRegions(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize)
{
    auto database = loadFeatureVectorsAndLabels("feature_vectors.csv");
    auto stdevs = calculateStandardDeviations(database);
    std::vector<std::string> orderedFeatureKeys = {
        "area", "percentFilled", "aspectRatio", "circularity", "compactness",
        "HuMoment 0", "HuMoment 1", "HuMoment 2", "HuMoment 3",
        "HuMoment 4", "HuMoment 5", "HuMoment 6"};

    cv::Mat invertedImg;
    cv::bitwise_not(binaryImage, invertedImg);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(invertedImg, labels, stats, centroids, 8, CV_32S);

    output = cv::Mat::zeros(originalImg.size(), CV_8UC3);

    cv::Point imageCenter = cv::Point(originalImg.cols / 2, originalImg.rows / 2);

    int centerAreaSize = std::min(originalImg.cols, originalImg.rows) / 3;
    cv::Rect centerArea(imageCenter.x - centerAreaSize, imageCenter.y - centerAreaSize, centerAreaSize * 2, centerAreaSize * 2);

    std::vector<int> objectsLabel;
    double minDistanceToCenter = std::numeric_limits<double>::max();
    std::string closestLabel = "Unknown";
    for (int i = 1; i < nLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area > minRegionSize)
        {
            cv::Point centroid = cv::Point(static_cast<int>(centroids.at<double>(i, 0)),
                                           static_cast<int>(centroids.at<double>(i, 1)));

            double distance = cv::norm(centroid - imageCenter);

            if (centerArea.contains(centroid))
            {
                objectsLabel.push_back(i);
            }
        }
    }

    std::vector<cv::Vec3b> colors(objectsLabel.size() + 1);

    for (size_t i = 0; i < objectsLabel.size(); i++)
    {
        colors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
    }

    for (size_t c = 0; c < objectsLabel.size(); c++)
    {
        int area = stats.at<int>(objectsLabel[c], cv::CC_STAT_AREA);
        int x = stats.at<int>(objectsLabel[c], cv::CC_STAT_LEFT);
        int y = stats.at<int>(objectsLabel[c], cv::CC_STAT_TOP);
        int width = stats.at<int>(objectsLabel[c], cv::CC_STAT_WIDTH);
        int height = stats.at<int>(objectsLabel[c], cv::CC_STAT_HEIGHT);

        cv::Mat mask = cv::Mat::zeros(originalImg.size(), CV_8UC1);
        for (int i = 0; i < mask.rows; i++)
        {
            uchar *maskptr = mask.ptr<uchar>(i);
            for (int j = 0; j < mask.cols; j++)
            {
                if (labels.at<int>(i, j) == objectsLabel[c])
                {
                    maskptr[j] = 255;
                }
            }
        }

        auto featureMap = computeFeatureVector(mask, area, width, height, labels, objectsLabel[c]);
        auto featureVector = convertFeatureMapToVector(featureMap, orderedFeatureKeys);
        drawFeatures(originalImg, featureMap, mask, x, y);

        double closestDistance = std::numeric_limits<double>::max();

        for (const auto &entry : database)
        {
            double distance = scaledEuclideanDistance(featureVector, entry.first, stdevs);
            if (distance < closestDistance)
            {
                closestDistance = distance;
                closestLabel = entry.second;
            }
        }

        cv::putText(originalImg, closestLabel, cv::Point(x, y - 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        for (int i = 0; i < originalImg.rows; i++)
        {
            for (int j = 0; j < originalImg.cols; j++)
            {
                if (labels.at<int>(i, j) == objectsLabel[c])
                {
                    output.at<cv::Vec3b>(i, j) = colors[c];
                }
            }
        }
    }
    return closestLabel;
}

/*
   Function: classifyWithKNN
   Purpose: Classifies a test vector using the k-Nearest Neighbors algorithm based on a dataset of feature vectors and labels.
   Arguments:
       testVector (const std::vector<double>&) : The feature vector to classify.
       database (const std::vector<std::pair<std::vector<double>, std::string>>&) : The dataset of feature vectors and labels.
       K (int) : The number of nearest neighbors to consider in the classification.
       stdevs (const std::vector<double>&) : The standard deviations of each feature, used for scaling distances.
   Returns: std::string : The predicted label for the test vector.
*/
std::string classifyWithKNN(const std::vector<double> &testVector,
                            const std::vector<std::pair<std::vector<double>, std::string>> &database,
                            int K,
                            const std::vector<double> &stdevs)
{
    std::vector<std::pair<double, std::string>> labeledDistances;
    for (const auto &entry : database)
    {
        double distance = scaledEuclideanDistance(testVector, entry.first, stdevs);
        labeledDistances.push_back({distance, entry.second});
    }

    std::sort(labeledDistances.begin(), labeledDistances.end());

    std::map<std::string, std::vector<double>> classDistances;
    for (int i = 0; i < K && i < labeledDistances.size(); ++i)
    {
        classDistances[labeledDistances[i].second].push_back(labeledDistances[i].first);
    }

    std::string closestClass = "";
    double smallestAvgDistance = std::numeric_limits<double>::max();
    for (const auto &pair : classDistances)
    {
        double avgDistance = std::accumulate(pair.second.begin(), pair.second.end(), 0.0) / pair.second.size();
        if (avgDistance < smallestAvgDistance)
        {
            smallestAvgDistance = avgDistance;
            closestClass = pair.first;
        }
    }

    return closestClass;
}

/*
   Function: classifyAndLabelRegionsKNN
   Purpose: Identifies, classifies, and labels regions in a binary image using the k-Nearest Neighbors algorithm. 
   Arguments:
       binaryImage (const cv::Mat&) : Input binary image.
       output (cv::Mat&) : The image where labeled regions will be visualized.
       originalImg (cv::Mat&) : The original image for reference.
       minRegionSize (int) : The minimum size of regions to consider for classification.
   Returns: std::string : The predicted label.
*/
std::string classifyAndLabelRegionsKNN(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize)
{
    auto database = loadFeatureVectorsAndLabels("feature_vectors.csv");
    auto stdevs = calculateStandardDeviations(database);
    std::vector<std::string> orderedFeatureKeys = {
        "area", "percentFilled", "aspectRatio", "circularity", "compactness",
        "HuMoment 0", "HuMoment 1", "HuMoment 2", "HuMoment 3",
        "HuMoment 4", "HuMoment 5", "HuMoment 6"};

    cv::Mat invertedImg;
    cv::bitwise_not(binaryImage, invertedImg);

    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(invertedImg, labels, stats, centroids, 8, CV_32S);

    output = cv::Mat::zeros(originalImg.size(), CV_8UC3);

    cv::Point imageCenter = cv::Point(originalImg.cols / 2, originalImg.rows / 2);

    int centerAreaSize = std::min(originalImg.cols, originalImg.rows) / 3;
    cv::Rect centerArea(imageCenter.x - centerAreaSize, imageCenter.y - centerAreaSize, centerAreaSize * 2, centerAreaSize * 2);

    std::vector<int> objectsLabel;
    double minDistanceToCenter = std::numeric_limits<double>::max();
    std::string label = "unknown";
    for (int i = 1; i < nLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area > minRegionSize)
        {
            cv::Point centroid = cv::Point(static_cast<int>(centroids.at<double>(i, 0)),
                                           static_cast<int>(centroids.at<double>(i, 1)));

            double distance = cv::norm(centroid - imageCenter);

            if (centerArea.contains(centroid))
            {
                objectsLabel.push_back(i);
            }
        }
    }

    std::vector<cv::Vec3b> colors(objectsLabel.size() + 1);

    for (size_t i = 0; i < objectsLabel.size(); i++)
    {
        colors[i] = cv::Vec3b(rand() & 255, rand() & 255, rand() & 255);
    }

    for (size_t c = 0; c < objectsLabel.size(); c++)
    {
        int area = stats.at<int>(objectsLabel[c], cv::CC_STAT_AREA);
        int x = stats.at<int>(objectsLabel[c], cv::CC_STAT_LEFT);
        int y = stats.at<int>(objectsLabel[c], cv::CC_STAT_TOP);
        int width = stats.at<int>(objectsLabel[c], cv::CC_STAT_WIDTH);
        int height = stats.at<int>(objectsLabel[c], cv::CC_STAT_HEIGHT);

        cv::Mat mask = cv::Mat::zeros(originalImg.size(), CV_8UC1);
        for (int i = 0; i < mask.rows; i++)
        {
            uchar *maskptr = mask.ptr<uchar>(i);
            for (int j = 0; j < mask.cols; j++)
            {
                if (labels.at<int>(i, j) == objectsLabel[c])
                {
                    maskptr[j] = 255;
                }
            }
        }

        auto featureMap = computeFeatureVector(mask, area, width, height, labels, objectsLabel[c]);
        auto featureVector = convertFeatureMapToVector(featureMap, orderedFeatureKeys);
        drawFeatures(originalImg, featureMap, mask, x, y);
        std::string label = classifyWithKNN(featureVector, database, 3, stdevs);
        cv::putText(originalImg, label, cv::Point(x, y - 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        for (int i = 0; i < originalImg.rows; i++)
        {
            for (int j = 0; j < originalImg.cols; j++)
            {
                if (labels.at<int>(i, j) == objectsLabel[c])
                {
                    output.at<cv::Vec3b>(i, j) = colors[c];
                }
            }
        }
    }
    return label;
}

/*
   Function: getEmbedding
   Purpose: Extracts feature embeddings from a specified region of an image using a pre-trained deep learning model. 
   Arguments:
       src (cv::Mat&) : The source image from which embeddings are extracted.
       embedding (cv::Mat&) : The output matrix where the extracted embeddings will be stored.
       bbox (cv::Rect&) : The bounding box specifying the region of interest in the source image.
       net (cv::dnn::Net&) : The pre-trained deep learning model used for feature extraction.
       debug (int) : Debug flag to enable visualization and printing of intermediate results.
   Returns: int : Status code (0 for success).
*/
int getEmbedding(cv::Mat &src, cv::Mat &embedding, cv::Rect &bbox, cv::dnn::Net &net, int debug)
{
    const int ORNet_size = 128;
    cv::Mat padImg;
    cv::Mat blob;

    cv::Mat roiImg = src(bbox);
    int top = bbox.height > 128 ? 10 : (128 - bbox.height) / 2 + 10;
    int left = bbox.width > 128 ? 10 : (128 - bbox.width) / 2 + 10;
    int bottom = top;
    int right = left;

    cv::copyMakeBorder(roiImg, padImg, top, bottom, left, right, cv::BORDER_CONSTANT, 0);
    cv::resize(padImg, padImg, cv::Size(128, 128));

    cv::dnn::blobFromImage(src,                              
                           blob,                             
                           (1.0 / 255.0) / 0.5,              
                           cv::Size(ORNet_size, ORNet_size), 
                           128,                              
                           false,                            
                           true,                             
                           CV_32F);                          

    net.setInput(blob);
    embedding = net.forward("onnx_node!/fc1/Gemm");

    if (debug)
    {
        cv::imshow("pad image", padImg);
        std::cout << embedding << std::endl;
        std::cout << "--Press any key to continue--";
        cv::waitKey(0);
    }

    return (0);
}