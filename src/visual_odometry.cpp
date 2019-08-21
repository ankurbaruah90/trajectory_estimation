#include "visual_odometry/vo_features.h"
#include <ros/ros.h>
#include <ros/package.h>
#include <dirent.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <tf/LinearMath/Matrix3x3.h>

#ifdef __MINGW32__
#include <sys/stat.h>
#endif

#define MAX_FRAME 250
#define MIN_NUM_FEAT 1500
#define SCALE 1

ros::NodeHandle *n_;
Mat rmat, tmat;
Mat prevImage;
vector<Point2f> prevFeatures;
Mat traj = Mat::zeros(600, 600, CV_8UC3);
double fx = 172.98992850734132;
double fy = 172.98992850734132;
double cx = 163.33639726024606;
double cy = 134.99537889030861;
double k1 = -0.027576733308582076;
double k2 = -0.006593578674675004;
double p1 = 0.0008566938165177085;
double p2 = -0.00030899587045247486;
Mat cameraMatrix = (Mat1d(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
Mat distortionCoefficients = (Mat1d(1, 4) << k1, k2, p1, p2);

// Code for sorting the string ----------------------------------------------------------
int strip_num(string str)
{
    size_t i = 0;
    for ( ; i < str.length(); i++ ){ if ( isdigit(str[i]) ) break; }
    str = str.substr(i + 2, str.length() - i );
    int id = atoi(str.c_str());
    return id;
}

bool custom_sort(string e1, string e2)
{
    int i1 = strip_num(e1);
    int i2 = strip_num(e2);
    return i1 < i2;
}
// --------------------------------------------------------------------------------------

// functions to read images from the folder data ------------------------------------------
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
//                printf("Found matching data file '%s'\n", ep->d_name);
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
// --------------------------------------------------------------------------------------

void imageCallBack(cv::Mat currImage)
{
    if(prevImage.empty())
        prevImage = currImage;

    vector<Point2f> currFeatures;
    if (prevFeatures.size() < MIN_NUM_FEAT)
        featureDetection(prevImage, prevFeatures);

    vector<uchar> status;
    bool feature_matches = featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);

    if(feature_matches)
    {

        Mat E, R, t, mask;
        E = findEssentialMat(currFeatures, prevFeatures, cameraMatrix, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, currFeatures, prevFeatures, cameraMatrix, R, t, mask);

        if(tmat.empty())
            tmat = t.clone();

        if(rmat.empty())
            rmat = R.clone();

        if((SCALE > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        {
            tmat = tmat + SCALE*(R*t);
            rmat = R*rmat;
        }

        //-------------------------------------------------------------------------------

        tf::Matrix3x3 tf3d;
        tf3d.setValue(rmat.at<double>(0,0), rmat.at<double>(0,1), rmat.at<double>(0,2),
              rmat.at<double>(1,0), rmat.at<double>(1,1), rmat.at<double>(1,2),
              rmat.at<double>(2,0), rmat.at<double>(2,1), rmat.at<double>(2,2));

        double roll, pitch, yaw;
        tf3d.getRPY(roll, pitch, yaw);

        cout << "Roll " << roll << " Pitch " << pitch << " Yaw " << yaw << endl;
        cout << "\nRotation " << R << "\nTranslation " << t << endl;

        //-------------------------------------------------------------------------------

        cv::Mat img_display = currImage.clone();
        for(size_t i1 = 0; i1 < status.size(); i1++)
        {
            if(status[i1] && (cv::norm(cv::Mat(prevFeatures[i1]), cv::Mat(currFeatures[i1])) <= int(MAX_INLIER_DIST)))
            {
                cv::circle(img_display, prevFeatures[i1], 2, cv::Scalar(0, 0, 255), 1);
                cv::circle(img_display, currFeatures[i1], 2, cv::Scalar(255, 0, 0), 1);
                cv::arrowedLine(img_display, prevFeatures[i1], currFeatures[i1], cv::Scalar(0, 255, 0), 2, 8, 0, 0.2);
            }
        }
        cv::imshow("feature_matches", img_display);
        cv::waitKey(1);

        prevFeatures.clear();
        prevFeatures = currFeatures;
    }

    prevImage = currImage.clone();

    int x = int(tmat.at<double>(0)) + traj.cols/2;
    int y = int(tmat.at<double>(2)) + traj.rows/2;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);
    cv::imshow( "Trajectory", traj );
    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "trajectory_estimation_node");
    n_ = new ros::NodeHandle;
//    ros::Rate rate(20);

    cv::Mat img,img_rectified;
    std::string path = ros::package::getPath("trajectory_estimation");
    static vector<string> images;
    static vector<string> validExtensions;
    validExtensions.push_back("jpg");
    validExtensions.push_back("png");
    validExtensions.push_back("ppm");
    static string samplesDir = path + "/data/mono/img/";
    getFilesInDirectory(samplesDir, images, validExtensions);
    std::sort(images.begin(),images.end(),custom_sort);

    int count = 0;
    for (vector<string>::iterator it = images.begin(); it != images.end(); ++it)
    {
        std::string fileName = *it;
        img = cv::imread(fileName);
        imshow("image",img);
        waitKey(1);
        undistort(img, img_rectified, cameraMatrix, distortionCoefficients);
        imageCallBack(img_rectified);
        cout << "\nProgress " << count++ << "%" << endl;
    }


//    ros::Subscriber image_subscriber = n_->subscribe("/camera_front/rgb/image_raw", 1, imageCallBack);
//    ros::Subscriber image_subscriber = n_->subscribe("/camera/color/image_raw", 1, imageCallBack);

//    while(ros::ok())
//    {
//        ros::spinOnce();
//        rate.sleep();
//    }

    delete n_;
    return 0;
}

