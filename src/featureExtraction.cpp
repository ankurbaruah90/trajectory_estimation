#include <iostream>
#include <math.h>
#include <ros/ros.h>
#include <ros/package.h>
#include <opencv/cv.h>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <fstream>
#include <string>
#include <unistd.h>
#include <dirent.h>
#include <lbp/lbp.cpp>

#ifdef __MINGW32__
#include <sys/stat.h>
#endif

#define LBP 0
#define HOG 1
#define RAW_DATA 0

using namespace cv;
using namespace std;

//static string posSamplesDir = "/home/ankur/Desktop/palletImages/pos/";
//static string posSamplesDir = "/home/ankur/Desktop/palletImages/MiscImages/pos/";
//static string negSamplesDir = "/home/ankur/Desktop/palletImages/neg/";
//static string negSamplesDir = "/home/ankur/Desktop/palletImages/Misc/Images/neg/";

static string toLowerCase(const string& in) {
    string t;
    for (string::const_iterator i = in.begin(); i != in.end(); ++i) {
        t += tolower(*i);
    }
    return t;
}

static void getFilesInDirectory(const string& dirName, vector<string>& fileNames, const vector<string>& validExtensions) {
    printf("Opening directory %s\n", dirName.c_str());
#ifdef __MINGW32__
        struct stat s;
#endif
    struct dirent* ep;
    size_t extensionLocation;
    DIR* dp = opendir(dirName.c_str());
    if (dp != NULL) {
        while ((ep = readdir(dp))) {
#ifdef __MINGW32__      
                        stat(ep->d_name, &s);
                        if (s.st_mode & S_IFDIR) {
                                continue;
                        }
#else
            if (ep->d_type & DT_DIR) {
                continue;
            }
#endif
            extensionLocation = string(ep->d_name).find_last_of("."); // Assume the last point marks beginning of extension like file.ext
            // Check if extension is matching the wanted ones
            string tempExt = toLowerCase(string(ep->d_name).substr(extensionLocation + 1));
            if (find(validExtensions.begin(), validExtensions.end(), tempExt) != validExtensions.end()) {
                printf("Found matching data file '%s'\n", ep->d_name);
                fileNames.push_back((string) dirName + ep->d_name);
            } else {
                printf("Found file does not match required file type, skipping: '%s'\n", ep->d_name);
            }
        }
        (void) closedir(dp);
    } else {
        printf("Error opening directory '%s'!\n", dirName.c_str());
    }
    return;
}

int main( int argc, char** argv )
{
    ros::init(argc, argv, "feature_extraction_node");
    std::string path = ros::package::getPath("pallet_detection");
    static vector<string> positiveTrainingImages;
    static vector<string> negativeTrainingImages;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("ppm");
    static string posSamplesDir = path + "/data/pos/";
    static string negSamplesDir = path + "/data/neg/";
    getFilesInDirectory(posSamplesDir, positiveTrainingImages, validExtensions);
    getFilesInDirectory(negSamplesDir, negativeTrainingImages, validExtensions);
    unsigned long overallSamples = positiveTrainingImages.size() + negativeTrainingImages.size();
    cout << "Images " << overallSamples << "\n";

    for (int currentFile = 0; currentFile < overallSamples; currentFile++)
    {
    cv::Mat img, img_gray, img_hog, img_lbp, hist;
const string imageFile = (currentFile < positiveTrainingImages.size() ? positiveTrainingImages.at(currentFile) : negativeTrainingImages.at(currentFile - positiveTrainingImages.size()));
    cout << "file " << imageFile << "\n";
    img = cv::imread(imageFile);
    vector <float> imageFeature;

    /*-----------Feature 1 - LBP histogram------------*/
#if LBP
    cvtColor(img, img_gray, CV_BGR2GRAY);
    GaussianBlur(img_gray, img_gray, Size(7,7), 5, 3, BORDER_CONSTANT);
    lbp::ELBP(img_gray, img_lbp, 1, 8);
    normalize(img_lbp, img_lbp, 0, 255, NORM_MINMAX, CV_8UC1);

    hist = cv::Mat::zeros(1, 256, CV_32SC1);
    for(int i = 0; i < img_lbp.rows; i++) {
        for(int j = 0; j < img_lbp.cols; j++) {
            int bin = img_lbp.at<uchar>(i,j);
            hist.at<int>(0,bin) += 1;
        }
    }

    for(int b = 0; b < 256; b++)
    {
        float const binVal = hist.at<int>(0,b);
        imageFeature.push_back(binVal);
    }

    cout << "1. " << imageFeature.size() << "\n";
#endif
    /*-------Feature 2 - HOG features-------*/
#if HOG  
    cvtColor(img, img_gray, CV_BGR2GRAY);
    img_hog = img_gray;
    resize(img_hog, img_hog, Size(32,16));
    HOGDescriptor hog;
    hog.winSize = Size(32,16);
    hog.blockSize = Size(16,16);
    hog.cellSize = Size(8,8);
    hog.blockStride = Size(8,8); 
    vector< float> descriptorValues;
    hog.compute(img_hog, descriptorValues);
    imageFeature.insert(imageFeature.end(), descriptorValues.begin(), descriptorValues.end());
    cout << "2. " << imageFeature.size() << "\n";
#endif

    int i;
    std::ofstream dataLog, resLog, libLog;
#if RAW_DATA
    dataLog.open("/home/ankur/Desktop/QtBetaVersions/palletDetection/data.csv", std::ofstream::out | std::ofstream::app);
    for (i = 0; i < imageFeature.size(); i++)
        dataLog << imageFeature[i] << ",";
    dataLog << "\n";
    dataLog.close();

    resLog.open("/home/ankur/Desktop/QtBetaVersions/palletDetection/label.csv", std::ofstream::out | std::ofstream::app);
    if (currentFile < positiveTrainingImages.size())
        resLog << "1";
    else
        resLog << "2";
    resLog << "\n";
    resLog.close();
#endif
    libLog.open(path + "/dataSet", std::ofstream::out | std::ofstream::app);
    if (currentFile < positiveTrainingImages.size())
        libLog << "1 ";
    else
        libLog << "2 ";
    for (i = 0; i < imageFeature.size(); i++)
        libLog << i+1 << ":" <<imageFeature[i] << " ";
    libLog << "\n";
    libLog.close();
    }

    return 0;
}

