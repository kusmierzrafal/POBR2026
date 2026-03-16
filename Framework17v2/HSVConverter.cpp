#include "HSVConverter.h"
#include <cmath>
#include <algorithm>

using namespace cv;
using namespace std;

Mat rgbToHsv(const Mat& src) {

    Mat hsvImage(src.rows, src.cols, src.type());

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {

            const unsigned char* srcPixel = src.ptr<unsigned char>(i) + j * 3;
            unsigned char* hsvPixel = hsvImage.ptr<unsigned char>(i) + j * 3;
            
            double b = srcPixel[0] / 255.0;
            double g = srcPixel[1] / 255.0;
            double r = srcPixel[2] / 255.0;

            double cmax = max({ r, g, b });
            double cmin = min({ r, g, b });
            double delta = cmax - cmin;

            double h = 0, s = 0, v = 0;

            if (delta == 0) {
                h = 0;
            }
            else if (cmax == r) {
                h = 60 * fmod(((g - b) / delta), 6);
            }
            else if (cmax == g) {
                h = 60 * (((b - r) / delta) + 2);
            }
            else if (cmax == b) {
                h = 60 * (((r - g) / delta) + 4);
            }

            if (h < 0) {
                h += 360;
            }

            if (cmax == 0) {
                s = 0;
            }
            else {
                s = delta / cmax;
            }

            v = cmax;

            hsvPixel[0] = static_cast<unsigned char>(h / 2);
            hsvPixel[1] = static_cast<unsigned char>(s * 255);
            hsvPixel[2] = static_cast<unsigned char>(v * 255);
        }
    }

    return hsvImage;
}