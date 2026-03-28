#ifndef LIDL_CONTENT_VERIFIER_H
#define LIDL_CONTENT_VERIFIER_H

#include <opencv2/opencv.hpp>

using namespace cv;

class LidlContentVerifier {
public:
    static Mat buildInnerMaskFromRedRing(const Mat& redRingROI, int erosionIterations);
    static bool verifyLidlInsideCircle(const Mat& hsvROI, const Mat& redRingROI);
    static Mat filterValidRings(const Mat& hsvImage, const Mat& redRingsMask);

private:
    static void zeroMat(Mat& img);
    static inline unsigned char get1(const Mat& img, int y, int x);
    static inline void set1(Mat& img, int y, int x, unsigned char value);

    static void erode3x3(const Mat& src, Mat& dst);

    static bool isYellowPixel(unsigned char h, unsigned char s, unsigned char v);
    static bool isBlueLetterPixel(unsigned char h, unsigned char s, unsigned char v);
    static bool isRedPixel(unsigned char h, unsigned char s, unsigned char v);
};

#endif