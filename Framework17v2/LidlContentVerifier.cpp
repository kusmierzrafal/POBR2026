#include "LidlContentVerifier.h"
#include <vector>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

struct VerifierPixel {
    int y, x;
};

struct VerifierComponent {
    vector<VerifierPixel> pixels;
    int minX, maxX, minY, maxY;
};

struct VerifierROI {
    int x, y;
    int width, height;
};

static void findVerifierComponents(const Mat& mask, vector<VerifierComponent>& comps)
{
    Mat visited(mask.rows, mask.cols, CV_8UC1);

    for (int y = 0; y < visited.rows; ++y) {
        unsigned char* row = visited.ptr<unsigned char>(y);
        for (int x = 0; x < visited.cols; ++x) row[x] = 0;
    }

    const int dy[8] = { -1,-1,-1, 0,0, 1,1,1 };
    const int dx[8] = { -1, 0, 1,-1,1,-1,0,1 };

    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            if (mask.ptr<unsigned char>(y)[x] != 255 || visited.ptr<unsigned char>(y)[x] != 0) continue;

            VerifierComponent comp;
            comp.minX = comp.maxX = x;
            comp.minY = comp.maxY = y;

            vector<VerifierPixel> queue;
            queue.push_back({ y, x });
            visited.ptr<unsigned char>(y)[x] = 1;

            for (size_t i = 0; i < queue.size(); ++i) {
                VerifierPixel p = queue[i];
                comp.pixels.push_back(p);

                if (p.x < comp.minX) comp.minX = p.x;
                if (p.x > comp.maxX) comp.maxX = p.x;
                if (p.y < comp.minY) comp.minY = p.y;
                if (p.y > comp.maxY) comp.maxY = p.y;

                for (int k = 0; k < 8; ++k) {
                    int ny = p.y + dy[k];
                    int nx = p.x + dx[k];

                    if (ny < 0 || ny >= mask.rows || nx < 0 || nx >= mask.cols) continue;
                    if (mask.ptr<unsigned char>(ny)[nx] != 255 || visited.ptr<unsigned char>(ny)[nx] != 0) continue;

                    visited.ptr<unsigned char>(ny)[nx] = 1;
                    queue.push_back({ ny, nx });
                }
            }

            comps.push_back(comp);
        }
    }
}

static VerifierROI expandCircleToROI(int centerX, int centerY, int radius, int imgWidth, int imgHeight)
{
    const int ROI_EXPANSION_NUMERATOR = 8;
    const int ROI_EXPANSION_DENOMINATOR = 5;

    int expandedRadius = (radius * ROI_EXPANSION_NUMERATOR + ROI_EXPANSION_DENOMINATOR / 2) / ROI_EXPANSION_DENOMINATOR;

    VerifierROI roi;
    roi.x = max(0, centerX - expandedRadius);
    roi.y = max(0, centerY - expandedRadius);

    int roiRight = min(imgWidth, centerX + expandedRadius);
    int roiBottom = min(imgHeight, centerY + expandedRadius);

    roi.width = roiRight - roi.x;
    roi.height = roiBottom - roi.y;
    return roi;
}

static Mat extractVerifierROI(const Mat& image, const VerifierROI& roi)
{
    Mat result(roi.height, roi.width, image.type());

    for (int y = 0; y < roi.height; ++y) {
        const unsigned char* srcRow = image.ptr<unsigned char>(roi.y + y);
        unsigned char* dstRow = result.ptr<unsigned char>(y);

        int channels = image.channels();
        for (int x = 0; x < roi.width; ++x) {
            for (int c = 0; c < channels; ++c) {
                dstRow[x * channels + c] = srcRow[(roi.x + x) * channels + c];
            }
        }
    }

    return result;
}

inline unsigned char LidlContentVerifier::get1(const Mat& img, int y, int x) {
    return img.ptr<unsigned char>(y)[x];
}

inline void LidlContentVerifier::set1(Mat& img, int y, int x, unsigned char value) {
    img.ptr<unsigned char>(y)[x] = value;
}

void LidlContentVerifier::zeroMat(Mat& img) {
    for (int y = 0; y < img.rows; ++y) {
        unsigned char* row = img.ptr<unsigned char>(y);
        int widthBytes = img.cols * img.channels();
        for (int x = 0; x < widthBytes; ++x) {
            row[x] = 0;
        }
    }
}

void LidlContentVerifier::erode3x3(const Mat& src, Mat& dst) {
    dst = Mat(src.rows, src.cols, CV_8UC1);
    zeroMat(dst);

    for (int y = 1; y < src.rows - 1; ++y) {
        for (int x = 1; x < src.cols - 1; ++x) {
            bool allWhite = true;

            for (int dy = -1; dy <= 1 && allWhite; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    if (get1(src, y + dy, x + dx) == 0) {
                        allWhite = false;
                        break;
                    }
                }
            }

            if (allWhite) {
                set1(dst, y, x, 255);
            }
        }
    }
}

bool LidlContentVerifier::isYellowPixel(unsigned char h, unsigned char s, unsigned char v) {
    return (h >= 18 && h <= 40 && s >= 90 && v >= 80);
}

bool LidlContentVerifier::isBlueLetterPixel(unsigned char h, unsigned char s, unsigned char v) {
    return (h >= 100 && h <= 135 && s >= 50 && v >= 20);
}

bool LidlContentVerifier::isRedPixel(unsigned char h, unsigned char s, unsigned char v) {
    bool hueRed = ((h >= 0 && h <= 12) || (h >= 170 && h <= 179));
    return hueRed && s >= 70 && v >= 35;
}

Mat LidlContentVerifier::buildInnerMaskFromRedRing(const Mat& redRingROI, int erosionIterations) {
    Mat visited(redRingROI.rows, redRingROI.cols, CV_8UC1);
    Mat innerMask(redRingROI.rows, redRingROI.cols, CV_8UC1);

    zeroMat(visited);
    zeroMat(innerMask);

    vector<VerifierPixel> queue;

    auto tryPush = [&](int y, int x) {
        if (y < 0 || y >= redRingROI.rows || x < 0 || x >= redRingROI.cols) return;
        if (get1(redRingROI, y, x) == 0 && get1(visited, y, x) == 0) {
            set1(visited, y, x, 1);
            queue.push_back({ y, x });
        }
        };

    for (int x = 0; x < redRingROI.cols; ++x) {
        tryPush(0, x);
        tryPush(redRingROI.rows - 1, x);
    }

    for (int y = 0; y < redRingROI.rows; ++y) {
        tryPush(y, 0);
        tryPush(y, redRingROI.cols - 1);
    }

    const int dy[4] = { -1, 1, 0, 0 };
    const int dx[4] = { 0, 0,-1, 1 };

    for (size_t i = 0; i < queue.size(); ++i) {
        VerifierPixel p = queue[i];

        for (int k = 0; k < 4; ++k) {
            int ny = p.y + dy[k];
            int nx = p.x + dx[k];

            if (ny < 0 || ny >= redRingROI.rows || nx < 0 || nx >= redRingROI.cols) continue;
            if (get1(redRingROI, ny, nx) != 0) continue;
            if (get1(visited, ny, nx) != 0) continue;

            set1(visited, ny, nx, 1);
            queue.push_back({ ny, nx });
        }
    }

    for (int y = 0; y < redRingROI.rows; ++y) {
        for (int x = 0; x < redRingROI.cols; ++x) {
            if (get1(redRingROI, y, x) == 0 && get1(visited, y, x) == 0) {
                set1(innerMask, y, x, 255);
            }
        }
    }

    Mat current = innerMask;
    for (int i = 0; i < erosionIterations; ++i) {
        Mat eroded;
        erode3x3(current, eroded);
        current = eroded;
    }

    return current;
}

bool LidlContentVerifier::verifyLidlInsideCircle(const Mat& hsvROI, const Mat& redRingROI) {
    Mat innerMask = buildInnerMaskFromRedRing(redRingROI, 2);

    int minX = innerMask.cols;
    int minY = innerMask.rows;
    int maxX = -1;
    int maxY = -1;
    int totalInside = 0;

    for (int y = 0; y < innerMask.rows; ++y) {
        for (int x = 0; x < innerMask.cols; ++x) {
            if (get1(innerMask, y, x) == 255) {
                totalInside++;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
            }
        }
    }

    if (totalInside < 40 || maxX <= minX || maxY <= minY) {
        return false;
    }

    int width = maxX - minX + 1;
    int height = maxY - minY + 1;
    int cx = (minX + maxX) / 2;
    int cy = (minY + maxY) / 2;

    int yellowCount = 0;
    int redCount = 0;

    int bandTotal = 0;
    int bandBlueCount = 0;

    int upperCenterTotal = 0;
    int upperCenterRedCount = 0;

    int centerTotal = 0;
    int centerRedCount = 0;

    for (int y = minY; y <= maxY; ++y) {
        const unsigned char* hsvRow = hsvROI.ptr<unsigned char>(y);

        for (int x = minX; x <= maxX; ++x) {
            if (get1(innerMask, y, x) != 255) continue;

            unsigned char h = hsvRow[3 * x + 0];
            unsigned char s = hsvRow[3 * x + 1];
            unsigned char v = hsvRow[3 * x + 2];

            bool yellow = isYellowPixel(h, s, v);
            bool blue = isBlueLetterPixel(h, s, v);
            bool red = isRedPixel(h, s, v);

            if (yellow) yellowCount++;
            if (red) redCount++;

            bool inMiddleBand =
                (x >= minX + (8 * width) / 100) &&
                (x <= maxX - (8 * width) / 100) &&
                (y >= cy - (16 * height) / 100) &&
                (y <= cy + (16 * height) / 100);

            if (inMiddleBand) {
                bandTotal++;
                if (blue) bandBlueCount++;
            }

            bool inUpperCenter =
                (x >= cx - (18 * width) / 100) &&
                (x <= cx + (18 * width) / 100) &&
                (y >= minY + (10 * height) / 100) &&
                (y <= minY + (40 * height) / 100);

            if (inUpperCenter) {
                upperCenterTotal++;
                if (red) upperCenterRedCount++;
            }

            bool inCenterRegion =
                (x >= cx - (22 * width) / 100) &&
                (x <= cx + (22 * width) / 100) &&
                (y >= cy - (22 * height) / 100) &&
                (y <= cy + (22 * height) / 100);

            if (inCenterRegion) {
                centerTotal++;
                if (red) centerRedCount++;
            }
        }
    }

    if (bandTotal < 10 || upperCenterTotal < 5 || centerTotal < 5) {
        return false;
    }

    double yellowFrac = (double)yellowCount / (double)totalInside;
    double redFrac = (double)redCount / (double)totalInside;
    double bandBlueFrac = (double)bandBlueCount / (double)bandTotal;
    double upperCenterRedFrac = (double)upperCenterRedCount / (double)upperCenterTotal;
    double centerRedFrac = (double)centerRedCount / (double)centerTotal;

    bool yellowOk = yellowFrac >= 0.55;
    bool blueOk = (bandBlueFrac >= 0.015 && bandBlueCount >= 4);
    bool redOk = (redFrac >= 0.002 && redCount >= 2);
    bool symbolOk =
        (upperCenterRedFrac >= 0.010 && upperCenterRedCount >= 1) ||
        (centerRedFrac >= 0.015 && centerRedCount >= 2);

    return yellowOk && blueOk && redOk && symbolOk;
}

Mat LidlContentVerifier::filterValidRings(const Mat& hsvImage, const Mat& redRingsMask) {
    vector<VerifierComponent> comps;
    findVerifierComponents(redRingsMask, comps);

    Mat filtered(redRingsMask.rows, redRingsMask.cols, CV_8UC1);
    zeroMat(filtered);

    int accepted = 0;

    for (size_t i = 0; i < comps.size(); ++i) {
        int width = comps[i].maxX - comps[i].minX + 1;
        int height = comps[i].maxY - comps[i].minY + 1;

        int centerX = comps[i].minX + width / 2;
        int centerY = comps[i].minY + height / 2;
        int radius = max(width, height) / 2;

        if (radius <= 10) continue;

        VerifierROI roi = expandCircleToROI(centerX, centerY, radius, hsvImage.cols, hsvImage.rows);

        Mat hsvROI = extractVerifierROI(hsvImage, roi);
        Mat redRingROI = extractVerifierROI(redRingsMask, roi);

        if (!verifyLidlInsideCircle(hsvROI, redRingROI)) {
            continue;
        }

        accepted++;
        for (size_t p = 0; p < comps[i].pixels.size(); ++p) {
            set1(filtered, comps[i].pixels[p].y, comps[i].pixels[p].x, 255);
        }
    }

    cout << "Zweryfikowano logotypow Lidl po srodku: " << accepted << endl;
    return filtered;
}