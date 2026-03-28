#include <opencv2/opencv.hpp>
#include "HSVConverter.h"
#include "RedRingDetector.h"
#include "LidlContentVerifier.h"
#include "BlueFrameDetector.h"

using namespace cv;
using namespace std;

int main()
{
    vector<string> image_paths = { "lidl_logo_1.png", "lidl_logo_2.png", "lidl_logo_3.png" };
    vector<Mat> images;

    for (const string& path : image_paths)
    {
        Mat img = imread(path);
        if (img.data == 0 || img.rows <= 0 || img.cols <= 0)
        {
            cout << "Nie można odczytać obrazu: " << path << endl;
            return -1;
        }
        images.push_back(img);
    }

    for (size_t i = 0; i < images.size(); ++i)
    {
        Mat hsv = rgbToHsv(images[i]);
        Mat redRings = RedRingDetector::detectRedRings(hsv);
        Mat verifiedRedRings = LidlContentVerifier::filterValidRings(hsv, redRings);
        Mat detectedLogos = BlueFrameDetector::drawDetectedLogos(images[i], hsv, verifiedRedRings);

        string logo_result_window_name = "Wykryte logo " + to_string(i + 1);
        string logo_result_filename = "detected_logos_lidl_logo_" + to_string(i + 1) + ".png";

        imshow(logo_result_window_name, detectedLogos);
        imwrite(logo_result_filename, detectedLogos);

        cout << "Zapisano: " << logo_result_filename << endl;
    }
    
    waitKey(0);
    return 0;
}
