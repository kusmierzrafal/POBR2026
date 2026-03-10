#include <iostream>
#include <opencv2/opencv.hpp>
#include "HSVConverter.h"

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    // Lista nazw plików wejściowych
    vector<string> image_paths = { "lidl_logo_1.png", "lidl_logo_2.png", "lidl_logo_3.png" };
    vector<Mat> images;
    vector<Mat> hsv_images;

    // Wczytanie obrazów
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

    // Konwersja obrazów do HSV
    for (size_t i = 0; i < images.size(); ++i)
    {
        cout << "Konwersja do HSV: " << image_paths[i] << endl;
        Mat hsv = rgbToHsv(images[i]);
        hsv_images.push_back(hsv);

        // Wyświetlenie TYLKO oryginalnego obrazu
        string original_window_name = "Oryginalny Obraz " + to_string(i + 1);
        imshow(original_window_name, images[i]);

        // Zapis obrazu HSV do pliku (ale go nie wyświetlamy!)
        string output_filename = "hsv_lidl_logo_" + to_string(i + 1) + ".png";
        imwrite(output_filename, hsv);
        cout << "Zapisano obraz HSV: " << output_filename << endl;
    }

    cout << "\nKonwersja zakończona. Obrazy HSV zapisane do plików." << endl;
    cout << "Wyświetlane są tylko oryginalne obrazy." << endl;
    cout << "Naciśnij dowolny klawisz, aby zamknąć..." << endl;
    
    waitKey(0);
    return 0;
}
