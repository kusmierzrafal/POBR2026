#include "ColorSegmentation.h"

using namespace cv;

Mat segmentRedColor(const Mat& hsvImage)
{
    Mat mask(hsvImage.rows, hsvImage.cols, CV_8UC1);
    
    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            unsigned char* maskPixel = mask.ptr<unsigned char>(i) + j;
            *maskPixel = 0;
        }
    }

    for (int i = 0; i < hsvImage.rows; i++)
    {
        for (int j = 0; j < hsvImage.cols; j++)
        {
            const unsigned char* hsvPixel = hsvImage.ptr<unsigned char>(i) + j * 3;
            int h = hsvPixel[0];
            int s = hsvPixel[1];
            int v = hsvPixel[2];

            bool isRed = false;
            if ((h >= 0 && h <= 10) && s >= 80 && v >= 40) isRed = true;
            if ((h >= 170 && h <= 179) && s >= 80 && v >= 40) isRed = true;

            unsigned char* maskPixel = mask.ptr<unsigned char>(i) + j;
            *maskPixel = isRed ? 255 : 0;
        }
    }

    return mask;
}

Mat segmentBlueColorLoose(const Mat& hsvImage)
{
    Mat mask(hsvImage.rows, hsvImage.cols, CV_8UC1);

    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            unsigned char* maskPixel = mask.ptr<unsigned char>(i) + j;
            *maskPixel = 0;
        }
    }

    for (int i = 0; i < hsvImage.rows; i++)
    {
        for (int j = 0; j < hsvImage.cols; j++)
        {
            const unsigned char* hsvPixel = hsvImage.ptr<unsigned char>(i) + j * 3;
            int h = hsvPixel[0];
            int s = hsvPixel[1];
            int v = hsvPixel[2];

            bool isBlue = false;
            if ((h >= 100 && h <= 130) && s >= 150 && s <= 255 && v >= 40 && v <= 255) isBlue = true;

            unsigned char* maskPixel = mask.ptr<unsigned char>(i) + j;
            *maskPixel = isBlue ? 255 : 0;
        }
    }

    return mask;
}

Mat segmentBlueColor(const Mat& hsvImage, int vMin)
{
    Mat mask(hsvImage.rows, hsvImage.cols, CV_8UC1);

    for (int i = 0; i < mask.rows; i++) {
        for (int j = 0; j < mask.cols; j++) {
            unsigned char* maskPixel = mask.ptr<unsigned char>(i) + j;
            *maskPixel = 0;
        }
    }

    for (int i = 0; i < hsvImage.rows; i++)
    {
        for (int j = 0; j < hsvImage.cols; j++)
        {
            const unsigned char* hsvPixel = hsvImage.ptr<unsigned char>(i) + j * 3;
            int h = hsvPixel[0];
            int s = hsvPixel[1];
            int v = hsvPixel[2];

            bool isBlue = false;
            if ((h >= 100 && h <= 130) && s >= 150 && s <= 255 && v >= vMin && v <= 255) isBlue = true;

            unsigned char* maskPixel = mask.ptr<unsigned char>(i) + j;
            *maskPixel = isBlue ? 255 : 0;
        }
    }

    return mask;
}
