#ifndef RED_RING_DETECTOR_H
#define RED_RING_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

struct Pixel {
    int y, x;
};

struct Component {
    vector<Pixel> pixels;
    int minX, maxX, minY, maxY;
    bool touchesBorder;
};

class RedRingDetector {
public:
    static Mat detectRedRings(const Mat& hsvImage);
    static Mat getRedColorMask(const Mat& hsvImage);

private:
    // Pipeline functions
    static Mat processEllipticalPipeline(const Mat& hsvImage, const Mat& strictMask, const Mat& looseMask, const vector<Component>& strictComps);
    static Mat processStandardPipeline(const Mat& hsvImage, const Mat& strictMask, const Mat& looseMask, const vector<Component>& strictComps);
    
    static void makeRedMasksFromHSV(const Mat& hsv, Mat& strictMask, Mat& looseMask);
    static Mat createEvenMoreLooseMask(const Mat& hsv);
    static inline bool isStrictRed(unsigned char H, unsigned char S, unsigned char V);
    static inline bool isLooseRed(unsigned char H, unsigned char S, unsigned char V);
    static inline bool isEvenMoreLooseRed(unsigned char H, unsigned char S, unsigned char V);
    
    static bool isReasonableSeed(const Component& c);
    static bool isEllipseLikeShape(const Component& c, bool printStats = false);
    static void copyComponentToMask(const Component& comp, Mat& dst, unsigned char value);
    static Mat expandMaskByNeighbors(const Mat& seedMask, const Mat& looseMask);
    static Mat cleanupNonCircularComponents(const Mat& inputMask);
    static Component expandComponentToLoose(const Component& strictComp, const Mat& looseMask);
    static void bridgeGaps(Mat& mask, const Component& comp);
    
    static void dilate3x3(const Mat& src, Mat& dst);
    static void erode3x3(const Mat& src, Mat& dst);
    static void close3x3(const Mat& src, Mat& dst, int iterations = 1);
    
    static void findComponents(const Mat& mask, vector<Component>& comps);
    static bool hasInternalHole(const Component& c, int& holeArea);
    static bool looksLikeClosedRedRing(const Component& c);
    static void buildResultMask(const vector<Component>& comps, Mat& out, int rows, int cols);
    
    // Helper functions for counting valid rings
    static int countValidRings(const vector<Component>& comps);                    // Standard strict criteria
    static int countValidRingsLooseCriteria(const vector<Component>& comps);      // Loose criteria for elliptical pipeline
    
    static inline unsigned char get1(const Mat& img, int y, int x);
    static inline void set1(Mat& img, int y, int x, unsigned char v);
    static void zeroMat(Mat& img);
};

#endif