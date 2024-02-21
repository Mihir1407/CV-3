cmake_minimum_required(VERSION 3.10)

project(FirstProject2)

set(OpenCV_DIR C:/opencv/build)

find_package(OpenCV REQUIRED)

# Added executable for imgDisplay.cp

add_executable(computeFeatures src/computeFeatures.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(computeFeatures ${OpenCV_LIBS})

add_executable(segmentImages src/segmentImages.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(segmentImages ${OpenCV_LIBS})

add_executable(morphologicalFilter src/morphologicalFilter.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(morphologicalFilter ${OpenCV_LIBS})

add_executable(threshold src/threshold.cpp src/kmeans.cpp src/objectRecogFunctions.cpp)
target_link_libraries(threshold ${OpenCV_LIBS})

# add_executable(ImgDisplay src/imgDisplay.cpp)
# target_link_libraries(ImgDisplay ${OpenCV_LIBS})

