#include "RedRingDetector.h"
#include <iostream>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

// Funkcje pomocnicze do operacji na pikselach
inline unsigned char RedRingDetector::get1(const Mat& img, int y, int x) {
    return img.ptr<unsigned char>(y)[x];
}

inline void RedRingDetector::set1(Mat& img, int y, int x, unsigned char v) {
    img.ptr<unsigned char>(y)[x] = v;
}

void RedRingDetector::zeroMat(Mat& img) {
    for (int y = 0; y < img.rows; ++y) {
        unsigned char* row = img.ptr<unsigned char>(y);
        for (int x = 0; x < img.cols; ++x) row[x] = 0;
    }
}

// Œcis³a detekcja czerwonego piksela - tylko pewna czerwieñ
inline bool RedRingDetector::isStrictRed(unsigned char H, unsigned char S, unsigned char V) {
    bool hueRed = (H <= 8) || (H >= 176);
    return hueRed && (S >= 110) && (V >= 45);
}

// LuŸna detekcja czerwonego piksela - standardowa
inline bool RedRingDetector::isLooseRed(unsigned char H, unsigned char S, unsigned char V) {
    bool hueRed = (H <= 12) || (H >= 172);
    return hueRed && (S >= 80) && (V >= 35);
}

// Bardzo luŸna detekcja czerwonego piksela - dla trudnych przypadków
inline bool RedRingDetector::isEvenMoreLooseRed(unsigned char H, unsigned char S, unsigned char V) {
    bool hueRed = (H <= 28) || (H >= 145);  // Jeszcze szerszy zakres H
    return hueRed && (S >= 40) && (V >= 10); // Bardzo niskie S i V
}

// Tworzenie trzech masek czerwonych pikseli - STRICT, LOOSE i EVEN_MORE_LOOSE
void RedRingDetector::makeRedMasksFromHSV(const Mat& hsv, Mat& strictMask, Mat& looseMask) {

    strictMask = Mat(hsv.rows, hsv.cols, 0);
    looseMask  = Mat(hsv.rows, hsv.cols, 0);
    
    zeroMat(strictMask);
    zeroMat(looseMask);

    for (int y = 0; y < hsv.rows; ++y) {
        const unsigned char* row = hsv.ptr<unsigned char>(y);
        unsigned char* outStrict = strictMask.ptr<unsigned char>(y);
        unsigned char* outLoose  = looseMask.ptr<unsigned char>(y);

        for (int x = 0; x < hsv.cols; ++x) {
            unsigned char H = row[3 * x + 0];
            unsigned char S = row[3 * x + 1];
            unsigned char V = row[3 * x + 2];

            outStrict[x] = isStrictRed(H, S, V) ? 255 : 0;
            outLoose[x]  = isLooseRed(H, S, V)  ? 255 : 0;
        }
    }
}

Mat RedRingDetector::createEvenMoreLooseMask(const Mat& hsv) {
    Mat evenMoreLooseMask = Mat(hsv.rows, hsv.cols, 0);
    zeroMat(evenMoreLooseMask);

    for (int y = 0; y < hsv.rows; ++y) {
        const unsigned char* row = hsv.ptr<unsigned char>(y);
        unsigned char* outEvenMoreLoose = evenMoreLooseMask.ptr<unsigned char>(y);

        for (int x = 0; x < hsv.cols; ++x) {
            unsigned char H = row[3 * x + 0];
            unsigned char S = row[3 * x + 1];
            unsigned char V = row[3 * x + 2];

            outEvenMoreLoose[x] = isEvenMoreLooseRed(H, S, V) ? 255 : 0;
        }
    }
    
    return evenMoreLooseMask;
}

// Znajdowanie spójnych komponentów (BFS)
void RedRingDetector::findComponents(const Mat& mask, vector<Component>& comps) {
    Mat visited(mask.rows, mask.cols, 0);
    zeroMat(visited);

    const int dy[8] = {-1,-1,-1, 0,0, 1,1,1};
    const int dx[8] = {-1, 0, 1,-1,1,-1,0,1};

    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            if (get1(mask, y, x) == 255 && get1(visited, y, x) == 0) {
                Component c;
                c.minX = c.maxX = x;
                c.minY = c.maxY = y;
                c.touchesBorder = false;

                vector<Pixel> queue;
                queue.push_back({y, x});
                set1(visited, y, x, 1);

                for (size_t i = 0; i < queue.size(); ++i) {
                    Pixel p = queue[i];
                    c.pixels.push_back(p);

                    if (p.x < c.minX) c.minX = p.x;
                    if (p.x > c.maxX) c.maxX = p.x;
                    if (p.y < c.minY) c.minY = p.y;
                    if (p.y > c.maxY) c.maxY = p.y;

                    if (p.x == 0 || p.x == mask.cols - 1 || p.y == 0 || p.y == mask.rows - 1)
                        c.touchesBorder = true;

                    for (int k = 0; k < 8; ++k) {
                        int ny = p.y + dy[k];
                        int nx = p.x + dx[k];

                        if (ny < 0 || ny >= mask.rows || nx < 0 || nx >= mask.cols)
                            continue;

                        if (get1(mask, ny, nx) == 255 && get1(visited, ny, nx) == 0) {
                            set1(visited, ny, nx, 1);
                            queue.push_back({ny, nx});
                        }
                    }
                }

                comps.push_back(c);
            }
        }
    }
}

// Sprawdzanie czy komponent ma wewnêtrzn¹ dziurê (flood fill od brzegu)
bool RedRingDetector::hasInternalHole(const Component& c, int& holeArea) {
    int h = (c.maxY - c.minY + 1) + 2;
    int w = (c.maxX - c.minX + 1) + 2;

    Mat local(h, w, 0);
    Mat visited(h, w, 0);
    zeroMat(local);
    zeroMat(visited);

    // Kopiowanie komponentu do lokalnego obrazu
    for (size_t i = 0; i < c.pixels.size(); ++i) {
        int ly = (c.pixels[i].y - c.minY) + 1;
        int lx = (c.pixels[i].x - c.minX) + 1;
        set1(local, ly, lx, 255);
    }

    vector<Pixel> queue;

    // Lambda do dodawania pikseli do flood fill
    auto tryPush = [&](int y, int x) {
        if (y < 0 || y >= h || x < 0 || x >= w) return;
        if (get1(local, y, x) == 0 && get1(visited, y, x) == 0) {
            set1(visited, y, x, 1);
            queue.push_back({y, x});
        }
    };

    // Rozpoczêcie flood fill od wszystkich brzegów
    for (int x = 0; x < w; ++x) {
        tryPush(0, x);
        tryPush(h - 1, x);
    }
    for (int y = 0; y < h; ++y) {
        tryPush(y, 0);
        tryPush(y, w - 1);
    }

    // BFS flood fill - 4-s¹siedztwo
    const int dy[4] = {-1, 1, 0, 0};
    const int dx[4] = { 0, 0,-1, 1};

    for (size_t i = 0; i < queue.size(); ++i) {
        Pixel p = queue[i];
        for (int k = 0; k < 4; ++k) {
            int ny = p.y + dy[k];
            int nx = p.x + dx[k];
            if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;

            if (get1(local, ny, nx) == 0 && get1(visited, ny, nx) == 0) {
                set1(visited, ny, nx, 1);
                queue.push_back({ny, nx});
            }
        }
    }

    // Liczenie pikseli niedostêpnych od brzegu (dziura wewnêtrzna)
    holeArea = 0;
    for (int y = 1; y < h - 1; ++y) {
        for (int x = 1; x < w - 1; ++x) {
            if (get1(local, y, x) == 0 && get1(visited, y, x) == 0) {
                holeArea++;
            }
        }
    }

    return holeArea > 0;
}

// Sprawdzanie czy komponent wygl¹da jak zamkniêta czerwona obrêcz
bool RedRingDetector::looksLikeClosedRedRing(const Component& c) {
    int width  = c.maxX - c.minX + 1;
    int height = c.maxY - c.minY + 1;
    int area   = (int)c.pixels.size();
    int bboxArea = width * height;

    if (c.touchesBorder) return false;
    if (width < 20 || height < 20) return false;
    if (area < 100) return false;

    double ratio = (double)width / (double)height;
    if (ratio < 0.45 || ratio > 1.8) return false;

    int holeArea = 0;
    if (!hasInternalHole(c, holeArea)) return false;

    double fill = (double)area / (double)bboxArea;

    if (holeArea < bboxArea / 6) return false;    // Dziura musi byæ wyraŸna
    if (fill < 0.03 || fill > 0.35) return false; // Ring nie mo¿e byæ ani za cienki, ani za pe³ny

    return true;
}

// Sprawdzenie czy komponent strict jest rozs¹dnym kandydatem na rozszerzenie
bool RedRingDetector::isReasonableSeed(const Component& c) {
    int width  = c.maxX - c.minX + 1;
    int height = c.maxY - c.minY + 1;
    int area   = (int)c.pixels.size();

    if (c.touchesBorder) return false;
    if (width < 6 || height < 6) return false;
    if (area < 12) return false;

    double ratio = (double)width / (double)height;
    if (ratio < 0.3 || ratio > 3.0) return false;

    return true;
}

// Tworzenie koñcowego obrazu z wykrytymi obrêczami
void RedRingDetector::buildResultMask(const vector<Component>& comps, Mat& out, int rows, int cols) {
    out = Mat(rows, cols, 0);
    zeroMat(out);

    for (size_t i = 0; i < comps.size(); ++i) {
        if (!looksLikeClosedRedRing(comps[i])) continue;

        for (size_t j = 0; j < comps[i].pixels.size(); ++j) {
            int y = comps[i].pixels[j].y;
            int x = comps[i].pixels[j].x;
            set1(out, y, x, 255);
        }
    }
}

// Dylatacja 3x3
void RedRingDetector::dilate3x3(const Mat& src, Mat& dst) {
    dst = Mat(src.rows, src.cols, 0);
    zeroMat(dst);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            bool white = false;

            for (int dy = -1; dy <= 1 && !white; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int ny = y + dy;
                    int nx = x + dx;

                    if (ny < 0 || ny >= src.rows || nx < 0 || nx >= src.cols)
                        continue;

                    if (get1(src, ny, nx) == 255) {
                        white = true;
                        break;
                    }
                }
            }

            if (white) set1(dst, y, x, 255);
        }
    }
}

// Erozja 3x3
void RedRingDetector::erode3x3(const Mat& src, Mat& dst) {
    dst = Mat(src.rows, src.cols, 0);
    zeroMat(dst);

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            bool allWhite = true;

            for (int dy = -1; dy <= 1 && allWhite; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    int ny = y + dy;
                    int nx = x + dx;

                    if (ny < 0 || ny >= src.rows || nx < 0 || nx >= src.cols) {
                        allWhite = false;
                        break;
                    }

                    if (get1(src, ny, nx) == 0) {
                        allWhite = false;
                        break;
                    }
                }
            }

            if (allWhite) set1(dst, y, x, 255);
        }
    }
}

// Closing 3x3
void RedRingDetector::close3x3(const Mat& src, Mat& dst, int iterations) {
    Mat a, b;

    a = Mat(src.rows, src.cols, 0);
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            set1(a, y, x, get1(src, y, x));
        }
    }

    for (int i = 0; i < iterations; ++i) {
        dilate3x3(a, b);
        erode3x3(b, a);
    }

    dst = a;
}

// Funkcja pomocnicza do liczenia prawid³owych obrêczy
int RedRingDetector::countValidRings(const vector<Component>& comps) {
    int validRings = 0;
    for (size_t i = 0; i < comps.size(); ++i) {
        if (looksLikeClosedRedRing(comps[i])) {
            validRings++;
        }
    }
    return validRings;
}

// G³ówna funkcja publiczna - AUTO-DETEKCJA na podstawie jakoœci wyniku
Mat RedRingDetector::detectRedRings(const Mat& hsvImage) {
    Mat strictMask, looseMask;
    makeRedMasksFromHSV(hsvImage, strictMask, looseMask);

    vector<Component> strictComps;
    findComponents(strictMask, strictComps);
    
    // Najpierw spróbuj standardowego pipeline'a
    Mat standardResult = processStandardPipeline(hsvImage, strictMask, looseMask, strictComps);
    
    // SprawdŸ jakoœæ wyniku standardowego
    vector<Component> standardComps;
    findComponents(standardResult, standardComps);
    int standardValidRings = countValidRings(standardComps);
    
    // Jeœli standard znalaz³ obrêcze, u¿yj go
    if (standardValidRings > 0) {
        cout << "Wykryto " << standardValidRings << " zamkniêtych obrêczy (pipeline standardowy)" << endl;
        return standardResult;
    }
    
    // Jeœli standard nie znalaz³ nic, spróbuj pipeline eliptyczny
    Mat ellipticalResult = processEllipticalPipeline(hsvImage, strictMask, looseMask, strictComps);
    
    // SprawdŸ jakoœæ wyniku pipeline'a eliptycznego
    vector<Component> ellipticalComps;
    findComponents(ellipticalResult, ellipticalComps);
    int ellipticalValidRings = countValidRingsLooseCriteria(ellipticalComps);
    
    cout << "Wykryto " << ellipticalValidRings << " zamkniêtych obrêczy (pipeline eliptyczny)" << endl;
    return ellipticalResult;
}

// Pipeline eliptyczny - u¿ywa eliptycznych komponentów i luŸniejszych kryteriów
Mat RedRingDetector::processEllipticalPipeline(const Mat& hsvImage, const Mat& strictMask, const Mat& looseMask, const vector<Component>& strictComps) {
    vector<Component> ellipticalComps;
    Mat ellipticalMask = Mat(hsvImage.rows, hsvImage.cols, 0);
    zeroMat(ellipticalMask);
    
    for (size_t i = 0; i < strictComps.size(); ++i) {
        if (isEllipseLikeShape(strictComps[i], false)) {
            ellipticalComps.push_back(strictComps[i]);
            copyComponentToMask(strictComps[i], ellipticalMask, 255);
        }
    }
    
    int ellipticalPixels = 0;
    for (int y = 0; y < ellipticalMask.rows; ++y) {
        for (int x = 0; x < ellipticalMask.cols; ++x) {
            if (get1(ellipticalMask, y, x) == 255) ellipticalPixels++;
        }
    }
    
    if (ellipticalPixels == 0) {
        Mat emptyMask = Mat(hsvImage.rows, hsvImage.cols, 0);
        zeroMat(emptyMask);
        return emptyMask;
    }
    
    // Rozszerzenie eliptycznych komponentów po loose mask
    Mat expandedEllipticalMask = expandMaskByNeighbors(ellipticalMask, looseMask);
    
    Mat evenMoreLooseMask = createEvenMoreLooseMask(hsvImage);
    Mat finalExpandedMask = expandMaskByNeighbors(expandedEllipticalMask, evenMoreLooseMask);
    
    Mat cleanedResult = cleanupNonCircularComponents(finalExpandedMask);
    
    return cleanedResult;
}

// Pipeline standardowy
Mat RedRingDetector::processStandardPipeline(const Mat& hsvImage, const Mat& strictMask, const Mat& looseMask, const vector<Component>& strictComps) {
    vector<Component> validSeeds;
    for (size_t i = 0; i < strictComps.size(); ++i) {
        if (isReasonableSeed(strictComps[i])) {
            validSeeds.push_back(strictComps[i]);
        }
    }

    vector<Component> expandedComps;
    for (size_t i = 0; i < validSeeds.size(); ++i) {
        Component expanded = expandComponentToLoose(validSeeds[i], looseMask);
        expandedComps.push_back(expanded);
    }

    Mat workingMask = Mat(hsvImage.rows, hsvImage.cols, 0);
    zeroMat(workingMask);
    for (size_t i = 0; i < expandedComps.size(); ++i) {
        // Narysuj komponent na masce roboczej
        for (size_t j = 0; j < expandedComps[i].pixels.size(); ++j) {
            Pixel p = expandedComps[i].pixels[j];
            set1(workingMask, p.y, p.x, 255);
        }
        
        // Zastosuj bridge gaps 2 razy
        bridgeGaps(workingMask, expandedComps[i]);
        bridgeGaps(workingMask, expandedComps[i]);
    }
    
    vector<Component> finalComps;
    findComponents(workingMask, finalComps);

    Mat result;
    buildResultMask(finalComps, result, hsvImage.rows, hsvImage.cols);
    
    return result;
}

// Sprawdzanie czy komponent ma kszta³t podobny do elipsy/okrêgu
bool RedRingDetector::isEllipseLikeShape(const Component& c, bool printStats) {
    int area = (int)c.pixels.size();
    int width = c.maxX - c.minX + 1;
    int height = c.maxY - c.minY + 1;

    if (area <= 0 || width <= 0 || height <= 0) {
        return false;
    }

    double aspect = (double)width / (double)height;
    double fill = (double)area / (double)(width * height);

    // Prosty model elipsy osiowo ustawionej wewn¹trz bbox
    double cx = 0.5 * (c.minX + c.maxX);
    double cy = 0.5 * (c.minY + c.maxY);
    double a = 0.5 * width;
    double b = 0.5 * height;

    double sumQ = 0.0;
    double sumQ2 = 0.0;

    for (size_t i = 0; i < c.pixels.size(); ++i) {
        double dx = c.pixels[i].x - cx;
        double dy = c.pixels[i].y - cy;
        double q = (dx * dx) / (a * a + 1e-9) + (dy * dy) / (b * b + 1e-9);
        sumQ += q;
        sumQ2 += q * q;
    }

    double meanQ = sumQ / (double)area;
    double varQ = sumQ2 / (double)area - meanQ * meanQ;
    if (varQ < 0.0) varQ = 0.0;
    double stdQ = sqrt(varQ);

    // Progi dobrane pod maskê Lidl - elipsy/okrêgi lub ich fragmenty
    if (area < 150) return false;                    // za ma³y
    if (width < 10 || height < 40) return false;    // za w¹ski/niski
    if (aspect < 0.15 || aspect > 1.25) return false; // z³e proporcje
    if (fill > 0.25) return false;                   // za gêsty (odrzuca pe³ne plamy)
    if (meanQ < 0.72 || meanQ > 1.20) return false; // nie pasuje do elipsy
    if (stdQ > 0.35) return false;                   // za rozrzucony

    return true;
}

// Kopiowanie komponentu do maski
void RedRingDetector::copyComponentToMask(const Component& comp, Mat& dst, unsigned char value) {
    for (size_t i = 0; i < comp.pixels.size(); ++i) {
        Pixel p = comp.pixels[i];
        set1(dst, p.y, p.x, value);
    }
}

// Iteracyjne rozszerzanie maski o 2 piksele do pe³nej konwergencji
Mat RedRingDetector::expandMaskByNeighbors(const Mat& seedMask, const Mat& looseMask) {

    Mat currentMask = Mat(seedMask.rows, seedMask.cols, 0);
    for (int y = 0; y < seedMask.rows; ++y) {
        for (int x = 0; x < seedMask.cols; ++x) {
            set1(currentMask, y, x, get1(seedMask, y, x));
        }
    }
    
    int iteration = 0;
    int totalAddedPixels = 0;
    
    while (true) {
        iteration++;

        Mat nextMask = Mat(currentMask.rows, currentMask.cols, 0);
        for (int y = 0; y < currentMask.rows; ++y) {
            for (int x = 0; x < currentMask.cols; ++x) {
                set1(nextMask, y, x, get1(currentMask, y, x));
            }
        }
        int addedInIteration = 0;
        
        // DYLATACJA o 2 piksele + filtrowanie przez loose mask
        for (int y = 0; y < currentMask.rows; ++y) {
            for (int x = 0; x < currentMask.cols; ++x) {
                // Jeœli piksel ju¿ jest bia³y, pomiñ
                if (get1(currentMask, y, x) == 255) {
                    continue;
                }
                
                // SprawdŸ czy piksel jest czerwony w loose mask
                if (get1(looseMask, y, x) != 255) {
                    continue;  // Nie jest czerwony - pomiñ
                }
                
                // SprawdŸ czy ma bia³ego s¹siada w odleg³oœci ? 2 (zasiêg 2 pikseli)
                bool hasWhiteNeighborInRange2 = false;
                
                for (int dy = -2; dy <= 2; ++dy) {
                    for (int dx = -2; dx <= 2; ++dx) {
                        // Pomiñ œrodkowy piksel
                        if (dy == 0 && dx == 0) continue;
                        
                        // U¿yj odleg³oœci Manhattan ? 2 dla bardziej naturalnego kszta³tu
                        if (abs(dy) + abs(dx) <= 2) {
                            int ny = y + dy;
                            int nx = x + dx;
                            
                            if (ny >= 0 && ny < currentMask.rows && nx >= 0 && nx < currentMask.cols) {
                                if (get1(currentMask, ny, nx) == 255) {
                                    hasWhiteNeighborInRange2 = true;
                                    break;
                                }
                            }
                        }
                    }
                    if (hasWhiteNeighborInRange2) break;
                }
                
                // Jeœli ma bia³ego s¹siada w zasiêgu 2 I jest czerwony w loose mask, dodaj go
                if (hasWhiteNeighborInRange2) {
                    set1(nextMask, y, x, 255);
                    addedInIteration++;
                    totalAddedPixels++;
                }
            }
        }
        
        // Jeœli nic siê nie zmieni³o, konwergencja osi¹gniêta
        if (addedInIteration == 0) {
            break;
        }
        
        currentMask = nextMask;  // Przygotuj do nastêpnej iteracji
        
        // Zabezpieczenie przed nieskoñczon¹ pêtl¹
        if (iteration > 50) {
            break;
        }
    }
    
    return currentMask;
}

// Rozszerzanie strict komponentu po loose masce (histereza)
Component RedRingDetector::expandComponentToLoose(const Component& strictComp, const Mat& looseMask) {
    Component expanded = strictComp; // Kopiuj podstawowe dane
    expanded.pixels.clear(); // Wyczyœæ listê pikseli
    
    Mat visited(looseMask.rows, looseMask.cols, 0);
    zeroMat(visited);
    
    vector<Pixel> queue;
    
    // Inicjalizacja: dodaj wszystkie strict piksele
    for (size_t i = 0; i < strictComp.pixels.size(); ++i) {
        Pixel p = strictComp.pixels[i];
        queue.push_back(p);
        set1(visited, p.y, p.x, 1);
    }
    
    // BFS rozszerzanie po loose pikselach
    const int dy[8] = {-1,-1,-1, 0,0, 1,1,1};
    const int dx[8] = {-1, 0, 1,-1,1,-1,0,1};
    
    for (size_t i = 0; i < queue.size(); ++i) {
        Pixel p = queue[i];
        expanded.pixels.push_back(p);
        
        // Aktualizuj bounding box
        if (p.x < expanded.minX) expanded.minX = p.x;
        if (p.x > expanded.maxX) expanded.maxX = p.x;
        if (p.y < expanded.minY) expanded.minY = p.y;
        if (p.y > expanded.maxY) expanded.maxY = p.y;
        
        // SprawdŸ czy dotyka brzegu
        if (p.x == 0 || p.x == looseMask.cols - 1 || p.y == 0 || p.y == looseMask.rows - 1)
            expanded.touchesBorder = true;
        
        // Rozszerz na s¹siadów w loose masce
        for (int k = 0; k < 8; ++k) {
            int ny = p.y + dy[k];
            int nx = p.x + dx[k];
            
            if (ny < 0 || ny >= looseMask.rows || nx < 0 || nx >= looseMask.cols)
                continue;
                
            if (get1(looseMask, ny, nx) == 255 && get1(visited, ny, nx) == 0) {
                set1(visited, ny, nx, 1);
                queue.push_back({ny, nx});
            }
        }
    }
    
    return expanded;
}

// Lokalne domykanie ma³ych szczelin w komponencie
void RedRingDetector::bridgeGaps(Mat& mask, const Component& comp) {
    // Pracuj tylko w obszarze komponentu + ma³y margines
    int margin = 3;
    int minX = max(0, comp.minX - margin);
    int maxX = min(mask.cols - 1, comp.maxX + margin);
    int minY = max(0, comp.minY - margin);
    int maxY = min(mask.rows - 1, comp.maxY + margin);
    
    Mat temp = Mat(mask.rows, mask.cols, 0);
    for (int y = 0; y < mask.rows; ++y) {
        for (int x = 0; x < mask.cols; ++x) {
            set1(temp, y, x, get1(mask, y, x));
        }
    }

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            if (get1(mask, y, x) == 255) continue; // Ju¿ bia³y
            
            // SprawdŸ czy mo¿na domkn¹æ lukê
            bool W = (x > 0) && (get1(mask, y, x-1) == 255);
            bool E = (x < mask.cols-1) && (get1(mask, y, x+1) == 255);
            bool N = (y > 0) && (get1(mask, y-1, x) == 255);
            bool S = (y < mask.rows-1) && (get1(mask, y+1, x) == 255);
            
            bool NW = (y > 0 && x > 0) && (get1(mask, y-1, x-1) == 255);
            bool NE = (y > 0 && x < mask.cols-1) && (get1(mask, y-1, x+1) == 255);
            bool SW = (y < mask.rows-1 && x > 0) && (get1(mask, y+1, x-1) == 255);
            bool SE = (y < mask.rows-1 && x < mask.cols-1) && (get1(mask, y+1, x+1) == 255);
            
            // Domknij typowe luki
            if ((W && E) || (N && S) || (NW && SE) || (NE && SW)) {
                set1(temp, y, x, 255);
            }
        }
    }
    
    mask = temp;
}

// Czyszczenie komponentów - zostaw tylko te które wygl¹daj¹ jak zamkniête okrêgi
Mat RedRingDetector::cleanupNonCircularComponents(const Mat& inputMask) {
    // ZnajdŸ wszystkie komponenty
    vector<Component> allComponents;
    findComponents(inputMask, allComponents);
    
    // Utwórz czyst¹ maskê
    Mat cleanMask = Mat(inputMask.rows, inputMask.cols, 0);
    zeroMat(cleanMask);
    int keptComponents = 0;
    
    for (size_t i = 0; i < allComponents.size(); ++i) {
        const Component& comp = allComponents[i];
        int width = comp.maxX - comp.minX + 1;
        int height = comp.maxY - comp.minY + 1;
        int area = (int)comp.pixels.size();
        int bboxArea = width * height;
        
        bool isValidRing = true;
        
        // 1. Podstawowe filtry
        if (comp.touchesBorder) {
            isValidRing = false;
        } else if (width < 15 || height < 15) {
            isValidRing = false;
        } else if (area < 80) {
            isValidRing = false;
        } else {
            // 2. Proporcje (bardziej luŸne dla elips w perspektywie)
            double ratio = (double)width / (double)height;
            if (ratio < 0.3 || ratio > 2.5) {
                isValidRing = false;
            } else {
                // 3. SprawdŸ czy ma dziurê wewnêtrzn¹
                int holeArea = 0;
                if (!hasInternalHole(comp, holeArea)) {
                    isValidRing = false;
                } else {
                    // 4. wymagania dotycz¹ce dziury i wype³nienia
                    double fill = (double)area / (double)bboxArea;
                    
                    if (holeArea < bboxArea / 10) {
                        isValidRing = false;
                    } else if (fill < 0.02 || fill > 0.5) {
                        isValidRing = false;
                    }
                }
            }
        }
        
        if (isValidRing) {
            // Skopiuj komponent do czystej maski
            copyComponentToMask(comp, cleanMask, 255);
            keptComponents++;
        } 
    }
    
    return cleanMask;
}

// Funkcja pomocnicza do liczenia prawid³owych obrêczy z luŸniejszymi kryteriami
int RedRingDetector::countValidRingsLooseCriteria(const vector<Component>& comps) {
    int validRings = 0;
    for (size_t i = 0; i < comps.size(); ++i) {
        const Component& comp = comps[i];
        int width = comp.maxX - comp.minX + 1;
        int height = comp.maxY - comp.minY + 1;
        int area = (int)comp.pixels.size();
        int bboxArea = width * height;
        
        bool isValidRing = true;
        
        if (comp.touchesBorder) {
            isValidRing = false;
        } else if (width < 15 || height < 15) {
            isValidRing = false;
        } else if (area < 80) {
            isValidRing = false;
        } else {
            double ratio = (double)width / (double)height;
            if (ratio < 0.3 || ratio > 2.5) {
                isValidRing = false;
            } else {
                int holeArea = 0;
                if (!hasInternalHole(comp, holeArea)) {
                    isValidRing = false;
                } else {
                    double fill = (double)area / (double)bboxArea;
                    if (holeArea < bboxArea / 10) {
                        isValidRing = false;
                    } else if (fill < 0.02 || fill > 0.5) {
                        isValidRing = false;
                    }
                }
            }
        }
        
        if (isValidRing) {
            validRings++;
        }
    }
    return validRings;
}