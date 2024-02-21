#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <numeric>
#include "kmeans.h"
#include "objectRecogFunctions.h"

// Utility function to convert a color image to grayscale
cv::Mat convertToGrayscale(const cv::Mat& src) {
    CV_Assert(src.type() == CV_8UC3);
    cv::Mat grayscale(src.rows, src.cols, CV_8UC1);
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(i, j);
            uchar gray = static_cast<uchar>(0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]);
            grayscale.at<uchar>(i, j) = gray;
        }
    }
    return grayscale;
}

// Utility function for Gaussian Blur
cv::Mat applyGaussianBlur(const cv::Mat& src, int kernelSize, double sigma) {
    cv::Mat blurred;
    cv::GaussianBlur(src, blurred, cv::Size(kernelSize, kernelSize), sigma, sigma);
    return blurred; // Using OpenCV's GaussianBlur for demonstration; replace with custom implementation if needed.
}

// Simple implementation of K-means for k=2 to find threshold
int kMeansThreshold(const cv::Mat& src, int maxIterations = 10) {
    std::vector<int> samples;
    for (int i = 0; i < src.rows; i += 4) {
        for (int j = 0; j < src.cols; j += 4) {
            samples.push_back(src.at<uchar>(i, j));
        }
    }

    // Initial guesses for centroids
    int centroid1 = 255, centroid2 = 0;
    for (int it = 0; it < maxIterations; ++it) {
        std::vector<int> cluster1, cluster2;
        // Assign samples to nearest centroid
        for (int val : samples) {
            if (std::abs(val - centroid1) < std::abs(val - centroid2)) {
                cluster1.push_back(val);
            } else {
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
cv::Mat applyThreshold(const cv::Mat& src, int threshold) {
    cv::Mat thresholded(src.size(), src.type());
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            thresholded.at<uchar>(i, j) = (src.at<uchar>(i, j) > threshold) ? 255 : 0;
        }
    }
    return thresholded;
}

// Main processing function
cv::Mat processFrameForThreshold(const cv::Mat& frame) {
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
cv::Mat applyErosion(const cv::Mat& src, int kernelSize) {
    cv::Mat eroded(src.size(), src.type(), cv::Scalar::all(0));
    int k = kernelSize / 2; // Kernel offset
    for (int i = k; i < src.rows - k; ++i) {
        for (int j = k; j < src.cols - k; ++j) {
            uchar min = 255;
            for (int ki = -k; ki <= k; ++ki) {
                for (int kj = -k; kj <= k; ++kj) {
                    uchar val = src.at<uchar>(i + ki, j + kj);
                    if (val < min) min = val;
                }
            }
            eroded.at<uchar>(i, j) = min == 255 ? 255 : 0;
        }
    }
    return eroded;
}

// Utility function for dilation
cv::Mat applyDilation(const cv::Mat& src, int kernelSize) {
    cv::Mat dilated(src.size(), src.type(), cv::Scalar::all(0));
    int k = kernelSize / 2; // Kernel offset
    for (int i = k; i < src.rows - k; ++i) {
        for (int j = k; j < src.cols - k; ++j) {
            uchar max = 0;
            for (int ki = -k; ki <= k; ++ki) {
                for (int kj = -k; kj <= k; ++kj) {
                    uchar val = src.at<uchar>(i + ki, j + kj);
                    if (val > max) max = val;
                }
            }
            dilated.at<uchar>(i, j) = max == 0 ? 0 : 255;
        }
    }
    return dilated;
}

// Function to clean up the thresholded image
cv::Mat cleanupBinaryImage(const cv::Mat& src, int erosionSize = 3, int dilationSize = 3) {
    // Apply erosion followed by dilation
    cv::Mat eroded = applyErosion(src, erosionSize);
    cv::Mat cleaned = applyDilation(eroded, dilationSize);
    return cleaned;
}

cv::Mat applyMorphologicalFilter(const cv::Mat& frame) {
    
    // Apply threshold
    cv::Mat thresholded = processFrameForThreshold(frame);
    
    // Clean up the binary image
    cv::Mat cleaned = cleanupBinaryImage(thresholded, 7, 7); // Example kernel sizes
    
    return cleaned;
}

struct UnionFind {
    std::vector<int> parent;

    UnionFind(int n) : parent(n) {
        for (int i = 0; i < n; ++i) parent[i] = i;
    }

    int find(int x) {
        if (parent[x] == x) return x;
        return parent[x] = find(parent[x]);
    }

    void unite(int x, int y) {
        x = find(x);
        y = find(y);
        if (x != y) parent[x] = y;
    }
};

int findMinNeighborLabel(const cv::Mat& labels, int i, int j) {
    std::vector<int> neighbors;
    // Check top and left (add top-right and left-bottom if 8-connectivity)
    if (i > 0 && labels.at<int>(i-1, j) > 0) neighbors.push_back(labels.at<int>(i-1, j));
    if (j > 0 && labels.at<int>(i, j-1) > 0) neighbors.push_back(labels.at<int>(i, j-1));
    if (neighbors.empty()) return 0; // No neighbors
    return *min_element(neighbors.begin(), neighbors.end());
}

cv::Mat applyConnectedComponents(const cv::Mat& src, int minSize) {
    cv::Mat labels = cv::Mat::zeros(src.size(), CV_32S);
    int nextLabel = 1;
    UnionFind uf(10000); // Arbitrary large size

    // First Pass
    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            if (src.at<uchar>(i, j) == 255) { // Assuming foreground is white
                int label = findMinNeighborLabel(labels, i, j);
                if (label == 0) {
                    label = nextLabel++;
                } else {
                    // Check for other neighbors and union labels if necessary
                    if (i > 0 && labels.at<int>(i-1, j) > 0) uf.unite(label, labels.at<int>(i-1, j));
                    if (j > 0 && labels.at<int>(i, j-1) > 0) uf.unite(label, labels.at<int>(i, j-1));
                }
                labels.at<int>(i, j) = label;
            }
        }
    }

    // Second Pass - Resolve labels
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            int label = labels.at<int>(i, j);
            if (label > 0) labels.at<int>(i, j) = uf.find(label);
        }
    }

    // Count sizes and remove small components
    std::map<int, int> labelSizes;
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            int label = labels.at<int>(i, j);
            if (label > 0) labelSizes[label]++;
        }
    }

    std::vector<cv::Vec3b> colors(nextLabel + 1);
    std::generate(colors.begin(), colors.end(), []() { return cv::Vec3b(rand() % 256, rand() % 256, rand() % 256); });

    cv::Mat output = cv::Mat::zeros(src.size(), CV_8UC3);
    for (int i = 0; i < labels.rows; ++i) {
        for (int j = 0; j < labels.cols; ++j) {
            int label = labels.at<int>(i, j);
            if (label > 0 && labelSizes[label] >= minSize) {
                output.at<cv::Vec3b>(i, j) = colors[label];
            }
        }
    }

    return output;
}

cv::Mat applyConnectedComponentsAndDisplayRegions(const cv::Mat& frame) {
    cv::Mat cleaned = applyMorphologicalFilter(frame); // Adjust kernel sizes as needed

    // Step 2: Apply Connected Components Labeling and Filter Small Regions
    // This step replaces the direct return of the cleaned image with the segmentation and visualization process.
    cv::Mat labeledRegions = applyConnectedComponents(cleaned, 50); // Example minSize = 50; adjust as needed

    return labeledRegions;
}

