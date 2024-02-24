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

// Utility function to convert a color image to grayscale
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

// Utility function for Gaussian Blur
cv::Mat applyGaussianBlur(const cv::Mat &src, int kernelSize, double sigma)
{
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(kernelSize, kernelSize), sigma, sigma);
    return blurred; // Using OpenCV's GaussianBlur for demonstration; replace with custom implementation if needed.
}

// Simple implementation of K-means for k=2 to find threshold
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

    // Initial guesses for centroids
    int centroid1 = 255, centroid2 = 0;
    for (int it = 0; it < maxIterations; ++it)
    {
        std::vector<int> cluster1, cluster2;
        // Assign samples to nearest centroid
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
        // Update centroids
        centroid1 = cluster1.empty() ? centroid1 : std::accumulate(cluster1.begin(), cluster1.end(), 0) / cluster1.size();
        centroid2 = cluster2.empty() ? centroid2 : std::accumulate(cluster2.begin(), cluster2.end(), 0) / cluster2.size();
    }
    return (centroid1 + centroid2) / 2; // Return the average of the two centroids as threshold
}

// Function to apply thresholding
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

// Main processing function
cv::Mat processFrameForThreshold(const cv::Mat &frame)
{
    // Convert to grayscale
    cv::Mat grayscale = convertToGrayscale(frame);

    // Apply Gaussian Blur
    cv::Mat blurred = applyGaussianBlur(grayscale, 5, 1.5); // Kernel size and sigma for Gaussian blur

    // Find optimal threshold using K-Means
    int thresholdValue = kMeansThreshold(blurred);

    // Apply threshold
    cv::Mat thresholded = applyThreshold(blurred, thresholdValue);

    return thresholded;
}

// Utility function for erosion
cv::Mat applyErosion(const cv::Mat &src, int kernelSize)
{
    cv::Mat eroded(src.size(), src.type(), cv::Scalar::all(0));
    int k = kernelSize / 2; // Kernel offset
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

// Utility function for dilation
cv::Mat applyDilation(const cv::Mat &src, int kernelSize)
{
    cv::Mat dilated(src.size(), src.type(), cv::Scalar::all(0));
    int k = kernelSize / 2; // Kernel offset
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

// Function to clean up the thresholded image
cv::Mat cleanupBinaryImage(const cv::Mat &src, int erosionSize = 3, int dilationSize = 3)
{
    // Apply erosion followed by dilation
    cv::Mat eroded = applyErosion(src, erosionSize);
    cv::Mat cleaned = applyDilation(eroded, dilationSize);
    return cleaned;
}

cv::Mat applyMorphologicalFilter(const cv::Mat &frame)
{

    // Apply threshold
    cv::Mat thresholded = processFrameForThreshold(frame);

    // Clean up the binary image
    cv::Mat cleaned = cleanupBinaryImage(thresholded, 7, 7); // Example kernel sizes

    return cleaned;
}

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
            { // Assuming foreground is white
                int label = findMinNeighborLabel(labels, i, j);
                if (label == 0)
                {
                    label = nextLabel++;
                }
                else
                {
                    // Check for other neighbors and union labels if necessary
                    if (i > 0 && labels.at<int>(i - 1, j) > 0)
                        uf.unite(label, labels.at<int>(i - 1, j));
                    if (j > 0 && labels.at<int>(i, j - 1) > 0)
                        uf.unite(label, labels.at<int>(i, j - 1));
                }
                labels.at<int>(i, j) = label;
            }
        }
    }

    // Second Pass - Resolve labels
    for (int i = 0; i < labels.rows; ++i)
    {
        for (int j = 0; j < labels.cols; ++j)
        {
            int label = labels.at<int>(i, j);
            if (label > 0)
                labels.at<int>(i, j) = uf.find(label);
        }
    }

    // Count sizes and remove small components
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

cv::Mat applyConnectedComponentsAndDisplayRegions(const cv::Mat &frame)
{
    cv::Mat cleaned = applyMorphologicalFilter(frame); // Adjust kernel sizes as needed

    // Step 2: Apply Connected Components Labeling and Filter Small Regions
    // This step replaces the direct return of the cleaned image with the segmentation and visualization process.
    cv::Mat labeledRegions = applyConnectedComponents(cleaned, 50); // Example minSize = 50; adjust as needed

    return labeledRegions;
}

cv::Mat displayConnectedComponents(const cv::Mat &img, int sizeThreshold = 100)
{
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(img, labels, stats, centroids, 8, CV_32S);

    static std::map<int, cv::Vec3b> labelColors; // Map to store colors for labels
    labelColors.clear();                         // Clear previous frame's data

    // Assign colors to labels, skipping small regions
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

void drawOrientedBoundingBox(const cv::Mat &mask, cv::Mat &output)
{
    std::vector<cv::Point> points;
    for (int y = 0; y < mask.rows; y++)
    {
        for (int x = 0; x < mask.cols; x++)
        {
            if (mask.at<uchar>(y, x) == 255)
            { // Assuming mask is a binary image
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

void drawAxisOfLeastCentralMoment(const cv::Mat &mask, cv::Mat &output)
{
    std::vector<cv::Point> points;
    for (int y = 0; y < mask.rows; y++)
    {
        for (int x = 0; x < mask.cols; x++)
        {
            if (mask.at<uchar>(y, x) == 255)
            { // Assuming mask is a binary image
                points.push_back(cv::Point(x, y));
            }
        }
    }

    if (points.empty())
        return; // Add a check to avoid PCA on empty data

    // Convert points to Mat of type CV_32F because PCA expects float type
    cv::Mat data = cv::Mat(points.size(), 2, CV_32F);
    for (size_t i = 0; i < points.size(); ++i)
    {
        data.at<float>(i, 0) = points[i].x;
        data.at<float>(i, 1) = points[i].y;
    }

    // Perform PCA
    cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

    // Accessing the PCA results correctly
    cv::Point2f center(pca.mean.at<float>(0), pca.mean.at<float>(1));
    cv::Vec2f eigenvector = pca.eigenvectors.at<float>(0); // Access the first eigenvector

    // Scale the eigenvector for visualization, adjusting the scale factor as needed
    cv::Point endpoint = center + cv::Point2f(eigenvector[0] * 100, eigenvector[1] * 100);

    // Draw the axis of least central moment
    cv::line(output, center, endpoint, cv::Scalar(255, 0, 0), 2);
}

std::unordered_map<std::string, double> computeFeatureVector(const cv::Mat &mask, const int area, const int width, const int height, const cv::Mat &labels, const int objectLabel)
{
    std::unordered_map<std::string, double> featureMap;

    double percentFilled = ((double)area / (width * height) * 100);
    double aspectRatio = (double)height / width;

    featureMap["area"] = area;
    featureMap["percentFilled"] = percentFilled;
    featureMap["aspectRatio"] = aspectRatio;

    // Calculating perimeter and other features
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    double perimeter = 0;
    if (!contours.empty())
    {
        perimeter = cv::arcLength(contours[0], true); // Assuming single object for simplicity
    }

    double circularity = (4 * M_PI * area) / (perimeter * perimeter);
    double compactness = sqrt((4 * area) / M_PI) / std::max(width, height);

    featureMap["circularity"] = circularity;
    featureMap["compactness"] = compactness;

    // Calculating Hu Moments for rotation and scale invariance
    cv::Moments objMoments = cv::moments(mask, true);
    double huMoments[7];
    cv::HuMoments(objMoments, huMoments);

    for (int i = 0; i < 7; i++)
    {
        featureMap["HuMoment " + std::to_string(i)] = -1 * copysign(1.0, huMoments[i]) * log10(std::abs(huMoments[i]));
    }

    return featureMap;
}

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

void findRegions(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize)
{
    cv::Mat invertedImg;
    cv::bitwise_not(binaryImage, invertedImg);

    // Perform connected components analysis on the inverted image
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(invertedImg, labels, stats, centroids, 8, CV_32S);

    output = cv::Mat::zeros(originalImg.size(), CV_8UC3);

    cv::Point imageCenter = cv::Point(originalImg.cols / 2, originalImg.rows / 2);

    int centerAreaSize = std::min(originalImg.cols, originalImg.rows) / 3;
    cv::Rect centerArea(imageCenter.x - centerAreaSize, imageCenter.y - centerAreaSize, centerAreaSize * 2, centerAreaSize * 2);
    cv::rectangle(originalImg, centerArea, cv::Scalar(255, 255, 0), 2);

    std::vector<int> objectsLabel;
    double minDistanceToCenter = std::numeric_limits<double>::max();

    // Iterate through all regions to find the centermost one
    for (int i = 1; i < nLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Only consider regions larger than the minimum size
        if (area > minRegionSize)
        {
            cv::Point centroid = cv::Point(static_cast<int>(centroids.at<double>(i, 0)),
                                           static_cast<int>(centroids.at<double>(i, 1)));

            // Compute the Euclidean distance from the centroid to the image center
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

    // Draw only the centermost region if one was found
    for (size_t c = 0; c < objectsLabel.size(); c++)
    {
        // Assuming computeFeatures is correctly adapted for cv:: namespace as well.
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
        // computeFeatures(originalImg, output, objectsLabel[c], labels, stats, centroids);
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

void saveFeatureVectorToCSV(const std::unordered_map<std::string, double> &featureMap, const std::string &label)
{
    std::ofstream file("feature_vectors.csv", std::ios::app); // Open in append mode
    if (file.is_open())
    {
        // Define the order of features as they should appear in the CSV
        std::vector<std::string> orderedFeatureKeys = {
            "area", "percentFilled", "aspectRatio", "circularity", "compactness",
            "HuMoment 0", "HuMoment 1", "HuMoment 2", "HuMoment 3",
            "HuMoment 4", "HuMoment 5", "HuMoment 6"};

        file << label; // Write label first

        for (const auto &key : orderedFeatureKeys)
        {
            if (featureMap.find(key) != featureMap.end())
            {                                      // Check if the key exists in the map
                file << "," << featureMap.at(key); // Write feature value preceded by comma
            }
            else
            {
                file << ","; // If the feature does not exist, write a placeholder (empty value)
            }
        }

        file << "\n"; // End of line
        file.close();
    }
    else
    {
        std::cerr << "Unable to open file for writing." << std::endl;
    }
}

void findRegionsAndStoreToCsv(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize)
{
    cv::Mat invertedImg;
    cv::bitwise_not(binaryImage, invertedImg);

    // Perform connected components analysis on the inverted image
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(invertedImg, labels, stats, centroids, 8, CV_32S);

    output = cv::Mat::zeros(originalImg.size(), CV_8UC3);

    cv::Point imageCenter = cv::Point(originalImg.cols / 2, originalImg.rows / 2);

    int centerAreaSize = std::min(originalImg.cols, originalImg.rows) / 3;
    cv::Rect centerArea(imageCenter.x - centerAreaSize, imageCenter.y - centerAreaSize, centerAreaSize * 2, centerAreaSize * 2);
    cv::rectangle(originalImg, centerArea, cv::Scalar(255, 255, 0), 2);

    std::vector<int> objectsLabel;
    double minDistanceToCenter = std::numeric_limits<double>::max();

    // Iterate through all regions to find the centermost one
    for (int i = 1; i < nLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Only consider regions larger than the minimum size
        if (area > minRegionSize)
        {
            cv::Point centroid = cv::Point(static_cast<int>(centroids.at<double>(i, 0)),
                                           static_cast<int>(centroids.at<double>(i, 1)));

            // Compute the Euclidean distance from the centroid to the image center
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

    // Draw only the centermost region if one was found
    for (size_t c = 0; c < objectsLabel.size(); c++)
    {
        // Assuming computeFeatures is correctly adapted for cv:: namespace as well.
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
        // Prompt for label input
        // Prompt for label input
        std::string label;
        std::cout << "Enter label for the detected object: ";
        std::cin >> label; // Consider adding error handling for cin

        // Save the feature vector and label to CSV
        saveFeatureVectorToCSV(featureMap, label);
        drawFeatures(output, featureMap, mask, x, y);
        // computeFeatures(originalImg, output, objectsLabel[c], labels, stats, centroids);
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

// Function to load feature vectors and their labels from a CSV file
std::vector<std::pair<std::vector<double>, std::string>> loadFeatureVectorsAndLabels(const std::string &fileName)
{
    std::vector<std::pair<std::vector<double>, std::string>> database;
    std::ifstream file(fileName); // Use the fileName argument to open the file
    std::string line;

    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << fileName << std::endl;
        return database; // Return an empty database if the file cannot be opened
    }

    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        std::string label;
        std::getline(iss, label, ','); // First entry is the label

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

// Helper function to convert featureMap to vector<double> in the specified order
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
            featureVector.push_back(0.0); // Consider how to handle missing values appropriately
        }
    }
    return featureVector;
}

std::vector<double> calculateStandardDeviations(const std::vector<std::pair<std::vector<double>, std::string>> &database)
{
    if (database.empty())
        return {};

    size_t numFeatures = database[0].first.size();
    std::vector<double> means(numFeatures, 0.0);
    std::vector<double> stdevs(numFeatures, 0.0);

    // Calculate means
    for (const auto &entry : database)
    {
        for (size_t i = 0; i < numFeatures; ++i)
        {
            means[i] += entry.first[i];
        }
    }
    for (double &mean : means)
        mean /= database.size();

    // Calculate standard deviations
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

// Modified distance function to include standard deviation scaling
double scaledEuclideanDistance(const std::vector<double> &vec1, const std::vector<double> &vec2, const std::vector<double> &stdevs)
{
    double distance = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i)
    {
        if (stdevs[i] > 0)
        { // Prevent division by zero
            double scaledDiff = (vec1[i] - vec2[i]) / stdevs[i];
            distance += scaledDiff * scaledDiff;
        }
    }
    return std::sqrt(distance);
}

std::string classifyAndLabelRegions(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize)
{
    // Load the database of feature vectors and labels
    auto database = loadFeatureVectorsAndLabels("feature_vectors.csv");
    auto stdevs = calculateStandardDeviations(database);
    // Specify the order of features
    std::vector<std::string> orderedFeatureKeys = {
        "area", "percentFilled", "aspectRatio", "circularity", "compactness",
        "HuMoment 0", "HuMoment 1", "HuMoment 2", "HuMoment 3",
        "HuMoment 4", "HuMoment 5", "HuMoment 6"};

    cv::Mat invertedImg;
    cv::bitwise_not(binaryImage, invertedImg);

    // Perform connected components analysis on the inverted image
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(invertedImg, labels, stats, centroids, 8, CV_32S);

    output = cv::Mat::zeros(originalImg.size(), CV_8UC3);

    cv::Point imageCenter = cv::Point(originalImg.cols / 2, originalImg.rows / 2);

    int centerAreaSize = std::min(originalImg.cols, originalImg.rows) / 3;
    cv::Rect centerArea(imageCenter.x - centerAreaSize, imageCenter.y - centerAreaSize, centerAreaSize * 2, centerAreaSize * 2);
    // cv::rectangle(originalImg, centerArea, cv::Scalar(255, 255, 0), 2);

    std::vector<int> objectsLabel;
    double minDistanceToCenter = std::numeric_limits<double>::max();
    // Placeholder for the nearest neighbor search
    std::string closestLabel = "Unknown";
    // Iterate through all regions to find the centermost one
    for (int i = 1; i < nLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Only consider regions larger than the minimum size
        if (area > minRegionSize)
        {
            cv::Point centroid = cv::Point(static_cast<int>(centroids.at<double>(i, 0)),
                                           static_cast<int>(centroids.at<double>(i, 1)));

            // Compute the Euclidean distance from the centroid to the image center
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

    // Draw only the centermost region if one was found
    for (size_t c = 0; c < objectsLabel.size(); c++)
    {
        // Assuming computeFeatures is correctly adapted for cv:: namespace as well.
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

        // Label the detected object on the output image
        // Assume x and y are the coordinates where you want to put the label
        cv::putText(originalImg, closestLabel, cv::Point(x, y - 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        // computeFeatures(originalImg, output, objectsLabel[c], labels, stats, centroids);
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

std::string classifyWithKNN(const std::vector<double> &testVector,
                            const std::vector<std::pair<std::vector<double>, std::string>> &database,
                            int K,
                            const std::vector<double> &stdevs)
{
    // Calculate distances
    std::vector<std::pair<double, std::string>> labeledDistances;
    for (const auto &entry : database)
    {
        double distance = scaledEuclideanDistance(testVector, entry.first, stdevs);
        labeledDistances.push_back({distance, entry.second});
    }

    // Sort by distance
    std::sort(labeledDistances.begin(), labeledDistances.end());

    // Aggregate distances by class, up to K nearest neighbors
    std::map<std::string, std::vector<double>> classDistances;
    for (int i = 0; i < K && i < labeledDistances.size(); ++i)
    {
        classDistances[labeledDistances[i].second].push_back(labeledDistances[i].first);
    }

    // Calculate average distance for each class and find the class with the smallest average distance
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

void classifyAndLabelRegionsKNN(const cv::Mat &binaryImage, cv::Mat &output, cv::Mat &originalImg, int minRegionSize)
{
    // Load the database of feature vectors and labels
    auto database = loadFeatureVectorsAndLabels("feature_vectors.csv");
    auto stdevs = calculateStandardDeviations(database);
    // Specify the order of features
    std::vector<std::string> orderedFeatureKeys = {
        "area", "percentFilled", "aspectRatio", "circularity", "compactness",
        "HuMoment 0", "HuMoment 1", "HuMoment 2", "HuMoment 3",
        "HuMoment 4", "HuMoment 5", "HuMoment 6"};

    cv::Mat invertedImg;
    cv::bitwise_not(binaryImage, invertedImg);

    // Perform connected components analysis on the inverted image
    cv::Mat labels, stats, centroids;
    int nLabels = cv::connectedComponentsWithStats(invertedImg, labels, stats, centroids, 8, CV_32S);

    output = cv::Mat::zeros(originalImg.size(), CV_8UC3);

    cv::Point imageCenter = cv::Point(originalImg.cols / 2, originalImg.rows / 2);

    int centerAreaSize = std::min(originalImg.cols, originalImg.rows) / 3;
    cv::Rect centerArea(imageCenter.x - centerAreaSize, imageCenter.y - centerAreaSize, centerAreaSize * 2, centerAreaSize * 2);
    // cv::rectangle(originalImg, centerArea, cv::Scalar(255, 255, 0), 2);

    std::vector<int> objectsLabel;
    double minDistanceToCenter = std::numeric_limits<double>::max();

    // Iterate through all regions to find the centermost one
    for (int i = 1; i < nLabels; i++)
    {
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Only consider regions larger than the minimum size
        if (area > minRegionSize)
        {
            cv::Point centroid = cv::Point(static_cast<int>(centroids.at<double>(i, 0)),
                                           static_cast<int>(centroids.at<double>(i, 1)));

            // Compute the Euclidean distance from the centroid to the image center
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

    // Draw only the centermost region if one was found
    for (size_t c = 0; c < objectsLabel.size(); c++)
    {
        // Assuming computeFeatures is correctly adapted for cv:: namespace as well.
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
        // Placeholder for the nearest neighbor search
        // Using classifyWithKNN to determine the label of the current region
        std::string label = classifyWithKNN(featureVector, database, 3, stdevs);
        // Label the detected object on the output image
        // Assume x and y are the coordinates where you want to put the label
        cv::putText(originalImg, label, cv::Point(x, y - 80), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
        // computeFeatures(originalImg, output, objectsLabel[c], labels, stats, centroids);
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

    cv::dnn::blobFromImage(src,                              // input image
                           blob,                             // output array
                           (1.0 / 255.0) / 0.5,              // scale factor
                           cv::Size(ORNet_size, ORNet_size), // resize the image to this
                           128,                              // subtract mean prior to scaling
                           false,                            // input is a single channel image
                           true,                             // center crop after scaling short side to size
                           CV_32F);                          // output depth/type

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