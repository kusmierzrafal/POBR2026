#ifndef HSV_CONVERTER_H
#define HSV_CONVERTER_H

#include <opencv2/opencv.hpp>

using namespace cv;

// Funkcja do konwersji obrazu z RGB na HSV
Mat rgbToHsv(const Mat& src);

#endif // HSV_CONVERTER_H