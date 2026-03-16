#include <iostream>
#include <opencv2/opencv.hpp>
#include "HSVConverter.h"
#include "RedRingDetector.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    vector<string> image_paths = { "lidl_logo_1.png", "lidl_logo_2.png", "lidl_logo_3.png" };
    vector<Mat> images;
    vector<Mat> hsv_images;

    for (const string& path : image_paths)
    {
        Mat img = imread(path);
        if (img.empty())
        {
            cout << "Nie można odczytać obrazu: " << path << endl;
            return -1;
        }
        images.push_back(img);
        cout << "Wczytano obraz: " << path << endl;
    }

    for (size_t i = 0; i < images.size(); ++i)
    {
        cout << "\nPrzetwarzanie obrazu: " << image_paths[i] << endl;
        
        Mat hsv = rgbToHsv(images[i]);
        hsv_images.push_back(hsv);
        
        Mat redRings = RedRingDetector::detectRedRings(hsv);

        string original_window_name = "Oryginalny " + to_string(i + 1);
        string result_window_name = "Czerwone obrącze " + to_string(i + 1);
        
        imshow(original_window_name, images[i]);
        imshow(result_window_name, redRings);

        string result_filename = "red_rings_lidl_logo_" + to_string(i + 1) + ".png";
        
        imwrite(result_filename, redRings);
        
        cout << "Zapisano wykryte obrącze: " << result_filename << endl;
    }

    cout << "\nPrzetwarzanie zakończone. Naciśnij dowolny klawisz..." << endl;
    waitKey(0);
    return 0;
}
