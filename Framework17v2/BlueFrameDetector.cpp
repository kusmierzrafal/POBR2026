#include "BlueFrameDetector.h"
#include "ColorSegmentation.h"
#include "LidlContentVerifier.h"
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

struct BlueMaskPixel {
    int y, x;
};

struct BlueMaskComponent {
    vector<BlueMaskPixel> pixels;
    int minX, maxX, minY, maxY;
};

int BlueFrameDetector::ROI_EXPANSION_NUMERATOR = 8;
int BlueFrameDetector::ROI_EXPANSION_DENOMINATOR = 5;

inline unsigned char BlueFrameDetector::get1(const Mat& img, int y, int x) {
    return img.ptr<unsigned char>(y)[x];
}

inline void BlueFrameDetector::set1(Mat& img, int y, int x, unsigned char v) {
    img.ptr<unsigned char>(y)[x] = v;
}

void BlueFrameDetector::zeroMat(Mat& img) {
    for (int y = 0; y < img.rows; ++y) {
        unsigned char* row = img.ptr<unsigned char>(y);
        int widthBytes = img.cols * img.channels();
        for (int x = 0; x < widthBytes; ++x) row[x] = 0;
    }
}

static void findBlueMaskComponents(const Mat& mask, vector<BlueMaskComponent>& comps) {
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

            BlueMaskComponent comp;
            comp.minX = comp.maxX = x;
            comp.minY = comp.maxY = y;

            vector<BlueMaskPixel> queue;
            queue.push_back({ y, x });
            visited.ptr<unsigned char>(y)[x] = 1;

            for (size_t i = 0; i < queue.size(); ++i) {
                BlueMaskPixel p = queue[i];
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

vector<DetectedCircle> BlueFrameDetector::findCirclesFromMask(const Mat& redRingsMask) {
    vector<DetectedCircle> circles;
    vector<BlueMaskComponent> comps;
    findBlueMaskComponents(redRingsMask, comps);

    for (size_t i = 0; i < comps.size(); ++i) {
        int width = comps[i].maxX - comps[i].minX + 1;
        int height = comps[i].maxY - comps[i].minY + 1;

        DetectedCircle circle;
        circle.centerX = comps[i].minX + width / 2;
        circle.centerY = comps[i].minY + height / 2;
        circle.radius = max(width, height) / 2;
        circle.boundingBox.x = comps[i].minX;
        circle.boundingBox.y = comps[i].minY;
        circle.boundingBox.width = width;
        circle.boundingBox.height = height;

        if (circle.radius > 10) circles.push_back(circle);
    }

    return circles;
}

BlueROI BlueFrameDetector::expandCircleToROI(int centerX, int centerY, int radius, int imgWidth, int imgHeight) {
    int expandedRadius = (radius * ROI_EXPANSION_NUMERATOR + ROI_EXPANSION_DENOMINATOR / 2) / ROI_EXPANSION_DENOMINATOR;

    BlueROI roi;
    roi.x = max(0, centerX - expandedRadius);
    roi.y = max(0, centerY - expandedRadius);
    int roiRight = min(imgWidth, centerX + expandedRadius);
    int roiBottom = min(imgHeight, centerY + expandedRadius);
    roi.width = roiRight - roi.x;
    roi.height = roiBottom - roi.y;
    return roi;
}

Mat BlueFrameDetector::extractROI(const Mat& image, const BlueROI& roi) {
    Mat result(roi.height, roi.width, image.type());
    zeroMat(result);

    int channels = image.channels();
    for (int y = 0; y < roi.height; ++y) {
        const unsigned char* srcRow = image.ptr<unsigned char>(roi.y + y);
        unsigned char* dstRow = result.ptr<unsigned char>(y);
        for (int x = 0; x < roi.width; ++x) {
            for (int c = 0; c < channels; ++c) {
                dstRow[x * channels + c] = srcRow[(roi.x + x) * channels + c];
            }
        }
    }

    return result;
}

void BlueFrameDetector::copyROIToImage(const Mat& roiMask, Mat& dst, int dstX, int dstY) {
    for (int y = 0; y < roiMask.rows; ++y) {
        for (int x = 0; x < roiMask.cols; ++x) {
            if (get1(roiMask, y, x) != 255) continue;
            int globalX = dstX + x;
            int globalY = dstY + y;
            if (globalX >= 0 && globalX < dst.cols && globalY >= 0 && globalY < dst.rows) {
                set1(dst, globalY, globalX, 255);
            }
        }
    }
}

int BlueFrameDetector::computeDynamicBlueVMin(const Mat& hsvROI, const Mat& looseBlueMask, int roiCenterX, int roiCenterY, int roiRadius) {
    int histogram[256];
    for (int i = 0; i < 256; ++i) histogram[i] = 0;

    int roiRadiusSq = roiRadius * roiRadius;
    int candidateCount = 0;

    for (int y = 0; y < hsvROI.rows; ++y) {
        const unsigned char* hsvRow = hsvROI.ptr<unsigned char>(y);
        for (int x = 0; x < hsvROI.cols; ++x) {
            int dx = x - roiCenterX;
            int dy = y - roiCenterY;
            if (dx * dx + dy * dy > roiRadiusSq) continue;
            if (get1(looseBlueMask, y, x) != 255) continue;
            histogram[hsvRow[3 * x + 2]]++;
            candidateCount++;
        }
    }

    if (candidateCount < 20) return 75;

    int targetCount = (candidateCount * 90 + 99) / 100;
    int cumulativeCount = 0;
    int vRef = 255;

    for (int v = 0; v <= 255; ++v) {
        cumulativeCount += histogram[v];
        if (cumulativeCount >= targetCount) {
            vRef = v;
            break;
        }
    }

    int dynamicVMin = vRef - 65;
    if (dynamicVMin < 75) dynamicVMin = 75;
    if (dynamicVMin > 190) dynamicVMin = 190;
    return dynamicVMin;
}

void BlueFrameDetector::keepOnlyComponentsTouchingMask(Mat& blueMask, const Mat& referenceMask) {
    Mat visited(blueMask.rows, blueMask.cols, CV_8UC1);
    Mat filteredMask(blueMask.rows, blueMask.cols, CV_8UC1);
    zeroMat(visited);
    zeroMat(filteredMask);

    const int dy[8] = { -1,-1,-1, 0,0, 1,1,1 };
    const int dx[8] = { -1, 0, 1,-1,1,-1,0,1 };
    const int maxDistance = 5;
    const int maxDistanceSq = maxDistance * maxDistance;

    for (int y = 0; y < blueMask.rows; ++y) {
        for (int x = 0; x < blueMask.cols; ++x) {
            if (get1(blueMask, y, x) != 255 || get1(visited, y, x) != 0) continue;

            vector<BlueMaskPixel> queue;
            vector<BlueMaskPixel> componentPixels;
            bool touchesReference = false;

            queue.push_back({ y, x });
            set1(visited, y, x, 1);

            for (size_t i = 0; i < queue.size(); ++i) {
                BlueMaskPixel p = queue[i];
                componentPixels.push_back(p);

                for (int ry = p.y - maxDistance; ry <= p.y + maxDistance && !touchesReference; ++ry) {
                    for (int rx = p.x - maxDistance; rx <= p.x + maxDistance; ++rx) {
                        if (ry < 0 || ry >= referenceMask.rows || rx < 0 || rx >= referenceMask.cols) continue;
                        int ddx = rx - p.x;
                        int ddy = ry - p.y;
                        if (ddx * ddx + ddy * ddy > maxDistanceSq) continue;
                        if (get1(referenceMask, ry, rx) == 255) {
                            touchesReference = true;
                            break;
                        }
                    }
                }

                for (int k = 0; k < 8; ++k) {
                    int ny = p.y + dy[k];
                    int nx = p.x + dx[k];
                    if (ny < 0 || ny >= blueMask.rows || nx < 0 || nx >= blueMask.cols) continue;
                    if (get1(blueMask, ny, nx) == 255 && get1(visited, ny, nx) == 0) {
                        set1(visited, ny, nx, 1);
                        queue.push_back({ ny, nx });
                    }
                }
            }

            if (touchesReference) {
                for (size_t i = 0; i < componentPixels.size(); ++i) {
                    set1(filteredMask, componentPixels[i].y, componentPixels[i].x, 255);
                }
            }
        }
    }

    blueMask = filteredMask;
}

bool BlueFrameDetector::estimateQuadrilateralFromMask(const Mat& blueMask, BluePoint& topLeft, BluePoint& topRight, BluePoint& bottomRight, BluePoint& bottomLeft) {
    bool foundAny = false;
    int minD1 = 0, maxD1 = 0, minD2 = 0, maxD2 = 0;

    for (int y = 0; y < blueMask.rows; ++y) {
        for (int x = 0; x < blueMask.cols; ++x) {
            if (get1(blueMask, y, x) != 255) continue;
            int d1 = x + y;
            int d2 = x - y;
            if (!foundAny) {
                foundAny = true;
                minD1 = maxD1 = d1;
                minD2 = maxD2 = d2;
            }
            else {
                if (d1 < minD1) minD1 = d1;
                if (d1 > maxD1) maxD1 = d1;
                if (d2 < minD2) minD2 = d2;
                if (d2 > maxD2) maxD2 = d2;
            }
        }
    }

    if (!foundAny) return false;

    long long tlX = 0, tlY = 0, tlCount = 0;
    long long trX = 0, trY = 0, trCount = 0;
    long long brX = 0, brY = 0, brCount = 0;
    long long blX = 0, blY = 0, blCount = 0;

    for (int y = 0; y < blueMask.rows; ++y) {
        for (int x = 0; x < blueMask.cols; ++x) {
            if (get1(blueMask, y, x) != 255) continue;
            int d1 = x + y;
            int d2 = x - y;
            if (d1 == minD1) { tlX += x; tlY += y; tlCount++; }
            if (d1 == maxD1) { brX += x; brY += y; brCount++; }
            if (d2 == maxD2) { trX += x; trY += y; trCount++; }
            if (d2 == minD2) { blX += x; blY += y; blCount++; }
        }
    }

    if (tlCount == 0 || trCount == 0 || brCount == 0 || blCount == 0) return false;

    topLeft.x = (int)(tlX / tlCount);
    topLeft.y = (int)(tlY / tlCount);
    topRight.x = (int)(trX / trCount);
    topRight.y = (int)(trY / trCount);
    bottomRight.x = (int)(brX / brCount);
    bottomRight.y = (int)(brY / brCount);
    bottomLeft.x = (int)(blX / blCount);
    bottomLeft.y = (int)(blY / blCount);
    return true;
}

void BlueFrameDetector::drawLine(Mat& image, BluePoint a, BluePoint b, unsigned char blue, unsigned char green, unsigned char red) {
    int x0 = a.x;
    int y0 = a.y;
    int x1 = b.x;
    int y1 = b.y;
    int dx = abs(x1 - x0);
    int sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0);
    int sy = y0 < y1 ? 1 : -1;
    int err = dx + dy;
    int thicknessRadius = 1;

    while (true) {
        for (int oy = -thicknessRadius; oy <= thicknessRadius; ++oy) {
            for (int ox = -thicknessRadius; ox <= thicknessRadius; ++ox) {
                int pxX = x0 + ox;
                int pxY = y0 + oy;
                if (pxX >= 0 && pxX < image.cols && pxY >= 0 && pxY < image.rows) {
                    unsigned char* px = image.ptr<unsigned char>(pxY) + pxX * image.channels();
                    if (image.channels() >= 3) {
                        px[0] = blue;
                        px[1] = green;
                        px[2] = red;
                    }
                    else {
                        px[0] = 255;
                    }
                }
            }
        }

        if (x0 == x1 && y0 == y1) break;

        int e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

Mat BlueFrameDetector::createBlueColorMask(const Mat& hsvROI, const Mat& redRingROI, int roiCenterX, int roiCenterY, int roiRadius, int innerRadiusX, int innerRadiusY) {

    Mat looseBlueMask = segmentBlueColorLoose(hsvROI);
    int dynamicVMin = computeDynamicBlueVMin(hsvROI, looseBlueMask, roiCenterX, roiCenterY, roiRadius);
    Mat blueMask = segmentBlueColor(hsvROI, dynamicVMin);

    Mat innerMask = LidlContentVerifier::buildInnerMaskFromRedRing(redRingROI, 1);
    int roiRadiusSq = roiRadius * roiRadius;

    for (int y = 0; y < blueMask.rows; ++y) {
        unsigned char* outMask = blueMask.ptr<unsigned char>(y);
        for (int x = 0; x < blueMask.cols; ++x) {
            int dx = x - roiCenterX;
            int dy = y - roiCenterY;

            bool outsideOuterRoi = dx * dx + dy * dy > roiRadiusSq;
            bool insideInnerMask = get1(innerMask, y, x) == 255;

            if (outsideOuterRoi || insideInnerMask) {
                outMask[x] = 0;
            }
        }
    }

    keepOnlyComponentsTouchingMask(blueMask, redRingROI);
    return blueMask;
}

BlueDetectionResult BlueFrameDetector::detectBlueMaskForCircle(const Mat& hsvImage, const Mat& redRingsMask, const DetectedCircle& circle) {
    BlueDetectionResult result;
    result.roi = expandCircleToROI(circle.centerX, circle.centerY, circle.radius, hsvImage.cols, hsvImage.rows);

    Mat hsvROI = extractROI(hsvImage, result.roi);
    Mat redRingROI = extractROI(redRingsMask, result.roi);
    int roiCenterX = circle.centerX - result.roi.x;
    int roiCenterY = circle.centerY - result.roi.y;
    int roiRadius = (circle.radius * ROI_EXPANSION_NUMERATOR + ROI_EXPANSION_DENOMINATOR / 2) / ROI_EXPANSION_DENOMINATOR;
    int innerRadiusX = circle.boundingBox.width / 2;
    int innerRadiusY = circle.boundingBox.height / 2;

    result.blueMask = createBlueColorMask(hsvROI, redRingROI, roiCenterX, roiCenterY, roiRadius, innerRadiusX, innerRadiusY);
    return result;
}

Mat BlueFrameDetector::detectBlueFrames(const Mat& originalImage, const Mat& hsvImage, const Mat& redRingsMask) {
    (void)originalImage;

    vector<DetectedCircle> circles = findCirclesFromMask(redRingsMask);
    Mat blueFramesMask(hsvImage.rows, hsvImage.cols, CV_8UC1);
    zeroMat(blueFramesMask);

    for (size_t i = 0; i < circles.size(); ++i) {
        BlueDetectionResult detection = detectBlueMaskForCircle(hsvImage, redRingsMask, circles[i]);
        copyROIToImage(detection.blueMask, blueFramesMask, detection.roi.x, detection.roi.y);
    }

    return blueFramesMask;
}

Mat BlueFrameDetector::drawDetectedLogos(const Mat& originalImage, const Mat& hsvImage, const Mat& redRingsMask) {
    Mat result(originalImage.rows, originalImage.cols, originalImage.type());

    for (int y = 0; y < originalImage.rows; ++y) {
        const unsigned char* srcRow = originalImage.ptr<unsigned char>(y);
        unsigned char* dstRow = result.ptr<unsigned char>(y);
        int widthBytes = originalImage.cols * originalImage.channels();
        for (int x = 0; x < widthBytes; ++x) dstRow[x] = srcRow[x];
    }

    vector<DetectedCircle> circles = findCirclesFromMask(redRingsMask);

    for (size_t i = 0; i < circles.size(); ++i) {
        BlueDetectionResult detection = detectBlueMaskForCircle(hsvImage, redRingsMask, circles[i]);

        BluePoint topLeft, topRight, bottomRight, bottomLeft;
        if (!estimateQuadrilateralFromMask(detection.blueMask, topLeft, topRight, bottomRight, bottomLeft)) continue;

        topLeft.x += detection.roi.x;
        topLeft.y += detection.roi.y;
        topRight.x += detection.roi.x;
        topRight.y += detection.roi.y;
        bottomRight.x += detection.roi.x;
        bottomRight.y += detection.roi.y;
        bottomLeft.x += detection.roi.x;
        bottomLeft.y += detection.roi.y;

        drawLine(result, topLeft, topRight, 0, 255, 0);
        drawLine(result, topRight, bottomRight, 0, 255, 0);
        drawLine(result, bottomRight, bottomLeft, 0, 255, 0);
        drawLine(result, bottomLeft, topLeft, 0, 255, 0);
    }

    return result;
}