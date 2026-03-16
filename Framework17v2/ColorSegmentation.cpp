#include "ColorSegmentation.h"

using namespace cv;

// Funkcja do segmentacji czerwonego koloru w przestrzeni HSV
Mat segmentRedColor(const Mat& hsvImage)
{
    Mat mask(hsvImage.rows, hsvImage.cols, hsvImage.type());
    
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
            
            int h = hsvPixel[0]; // Hue (0-179)
            int s = hsvPixel[1]; // Saturation (0-255)
            int v = hsvPixel[2]; // Value (0-255)

            bool isRed = false;

            // Sprawdzanie pierwszego zakresu czerwieni
            if ((h >= 0 && h <= 10) && s >= 80 && v >= 40)
            {
                isRed = true;
            }
            
            // Sprawdzanie drugiego zakresu czerwieni
            if ((h >= 170 && h <= 179) && s >= 80 && v >= 40)
            {
                isRed = true;
            }

            // Ustawianie maski: bia³y dla czerwonych pikseli, czarny dla reszty
            unsigned char* maskPixel = mask.ptr<unsigned char>(i) + j;
            if (isRed)
            {
                *maskPixel = 255; // Bia³y
            }
            else
            {
                *maskPixel = 0;   // Czarny
            }
        }
    }

    return mask;
}