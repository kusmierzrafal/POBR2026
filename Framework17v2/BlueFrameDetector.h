#ifndef BLUE_FRAME_DETECTOR_H
#define BLUE_FRAME_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

struct BlueROI {
    int x, y;
    int width, height;
};

struct DetectedCircle {
    int centerX, centerY;
    int radius;
    BlueROI boundingBox;
};

struct BluePoint {
    int x, y;
};

struct BlueDetectionResult {
    BlueROI roi;
    Mat blueMask;
};

class BlueFrameDetector {
public:
    static Mat detectBlueFrames(const Mat& originalImage, const Mat& hsvImage, const Mat& redRingsMask);
    static Mat drawDetectedLogos(const Mat& originalImage, const Mat& hsvImage, const Mat& redRingsMask);
    
private:
    static int ROI_EXPANSION_NUMERATOR;
    static int ROI_EXPANSION_DENOMINATOR;
    
    static vector<DetectedCircle> findCirclesFromMask(const Mat& redRingsMask);
    static BlueDetectionResult detectBlueMaskForCircle(const Mat& hsvImage, const Mat& redRingsMask, const DetectedCircle& circle);
    static Mat createBlueColorMask(const Mat& hsvROI, const Mat& redRingROI, int roiCenterX, int roiCenterY, int roiRadius, int innerRadiusX, int innerRadiusY);
    static int computeDynamicBlueVMin(const Mat& hsvROI, const Mat& looseBlueMask, int roiCenterX, int roiCenterY, int roiRadius);
    static void keepOnlyComponentsTouchingMask(Mat& blueMask, const Mat& referenceMask);
    static bool estimateQuadrilateralFromMask(const Mat& blueMask, BluePoint& topLeft, BluePoint& topRight, BluePoint& bottomRight, BluePoint& bottomLeft);
    static void drawLine(Mat& image, BluePoint a, BluePoint b, unsigned char blue, unsigned char green, unsigned char red);
    static BlueROI expandCircleToROI(int centerX, int centerY, int radius, int imgWidth, int imgHeight);
    static Mat extractROI(const Mat& image, const BlueROI& roi);
    static void copyROIToImage(const Mat& roiMask, Mat& dst, int dstX, int dstY);
    static void zeroMat(Mat& img);
    static inline unsigned char get1(const Mat& img, int y, int x);
    static inline void set1(Mat& img, int y, int x, unsigned char v);
};

#endif // BLUE_FRAME_DETECTOR_H
