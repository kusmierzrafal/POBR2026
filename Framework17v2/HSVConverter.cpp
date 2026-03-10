#include "HSVConverter.h"
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

// Funkcja do konwersji obrazu z RGB na HSV
Mat rgbToHsv(const Mat& src)
{
    Mat hsvImage = Mat::zeros(src.rows, src.cols, src.type());

    for (int i = 0; i < src.rows; i++)
    {
        for (int j = 0; j < src.cols; j++)
        {
            Vec3b pixel = src.at<Vec3b>(i, j);
            double b = pixel[0] / 255.0;
            double g = pixel[1] / 255.0;
            double r = pixel[2] / 255.0;

            double cmax = max({ r, g, b });
            double cmin = min({ r, g, b });
            double delta = cmax - cmin;

            double h = 0, s = 0, v = 0;

            // Obliczanie Hue (odcieñ)
            if (delta == 0)
            {
                h = 0;
            }
            else if (cmax == r)
            {
                h = 60 * fmod(((g - b) / delta), 6);
            }
            else if (cmax == g)
            {
                h = 60 * (((b - r) / delta) + 2);
            }
            else if (cmax == b)
            {
                h = 60 * (((r - g) / delta) + 4);
            }

            if (h < 0)
            {
                h += 360;
            }

            // Obliczanie Saturation (nasycenie)
            if (cmax == 0)
            {
                s = 0;
            }
            else
            {
                s = delta / cmax;
            }

            // Obliczanie Value (wartoœæ/jasnoœæ)
            v = cmax;

            hsvImage.at<Vec3b>(i, j)[0] = static_cast<uchar>(h / 2); // Skalowanie H do 0-179
            hsvImage.at<Vec3b>(i, j)[1] = static_cast<uchar>(s * 255);
            hsvImage.at<Vec3b>(i, j)[2] = static_cast<uchar>(v * 255);
        }
    }

    return hsvImage;
}