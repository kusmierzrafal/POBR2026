#ifndef COLOR_SEGMENTATION_H
#define COLOR_SEGMENTATION_H

#include <opencv2/opencv.hpp>

using namespace cv;

Mat segmentRedColor(const Mat& hsvImage);
Mat segmentBlueColorLoose(const Mat& hsvImage);
Mat segmentBlueColor(const Mat& hsvImage, int vMin);

#endif // COLOR_SEGMENTATION_H