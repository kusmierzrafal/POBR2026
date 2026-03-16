#ifndef COLOR_SEGMENTATION_H
#define COLOR_SEGMENTATION_H

#include <opencv2/opencv.hpp>

using namespace cv;

// Funkcja do segmentacji czerwonego koloru w przestrzeni HSV
Mat segmentRedColor(const Mat& hsvImage);

#endif // COLOR_SEGMENTATION_H