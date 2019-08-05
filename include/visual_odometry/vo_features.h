#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include "OpenSURFcpp/src/surflib.h"

#include <iostream>
#include <ctype.h>
#include <algorithm> // for copy
#include <iterator> // for ostream_iterator
#include <vector>
#include <ctime>
#include <sstream>
#include <fstream>
#include <string>

#define USE_ORB 0
#define USE_OPEN_SURF 1
#define MAX_INLIER_DIST 15
#define MIN_POINTS 20

using namespace cv;
using namespace std;

bool featureTracking(Mat img_1, Mat img_2, vector<Point2f>& points1, vector<Point2f>& points2, vector<uchar>& status)	{

    //this function automatically gets rid of points for which tracking fails
    vector<float> err;
    Size winSize=Size(21,21);
    calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err, winSize);

    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for( int i=0; i<status.size(); i++)
    {
        Point2f pt = points2.at(i- indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)
             || (cv::norm(cv::Mat(points1[i]), cv::Mat(points2[i])) >= int(MAX_INLIER_DIST)))
        {
            if((pt.x<0) || (pt.y<0))
            {
                status.at(i) = 0;
            }
            points1.erase (points1.begin() + (i - indexCorrection));
            points2.erase (points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }

    if(points2.size() < MIN_POINTS)
        return false;

    return true;
}

void featureDetection(Mat img_1, vector<Point2f>& points1)
{
    cv::Mat binary, binary_tmp;
    cv::GaussianBlur(img_1, binary_tmp, cv::Size(1,1), cv::BORDER_DEFAULT);
    cv::addWeighted(binary_tmp, 1.5, img_1, -0.5, 0, binary);
    cv::Mat keypoints_1_desc;
    std::vector<cv::KeyPoint> keypoints_1;

#if USE_ORB
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(1000, 1.2f, 4, 31, 0, 2, cv::ORB::HARRIS_SCORE, 24, 20);
    detector->detect(binary, keypoints_1, cv::noArray());
    detector->compute(binary, keypoints_1, keypoints_1_desc);
#elif USE_OPEN_SURF
    int count = 0;
    IplImage *img = new IplImage(binary);
    std::vector<Ipoint> opensurf_keyp;
    surfDetDes(img, opensurf_keyp, true, 8, 100, INIT_SAMPLE, 0.0);
    keypoints_1.resize(opensurf_keyp.size());
    keypoints_1_desc = cv::Mat(cv::Size(64,opensurf_keyp.size()), CV_32FC1, 0.0);
    for(std::vector <Ipoint>::iterator it = opensurf_keyp.begin(); it != opensurf_keyp.end(); ++it){
        keypoints_1[count].pt.x = (*it).x;
        keypoints_1[count].pt.y = (*it).y;
        keypoints_1[count].angle = (*it).orientation;
        cv::Mat tmp = cv::Mat(cv::Size(64,1), CV_32FC1, (*it).descriptor);
        for(int i = 0; i < tmp.cols; i++)
            keypoints_1_desc.at<float>(count, i) = tmp.at<float>(0, i);
        count++;
    }
    delete img;
#else
    int fast_threshold = 5;
    bool nonmaxSuppression = true;
    FAST(img_1, keypoints_1, fast_threshold, nonmaxSuppression);
#endif

    KeyPoint::convert(keypoints_1, points1, vector<int>());

    cv::Mat img_display = img_1.clone();
    cv::Mat pointmat;
    cv::drawKeypoints(img_display, keypoints_1, pointmat, cv::Scalar(0, 0, 255));
    cv::imshow("keypoints", pointmat);
    cv::waitKey(1);

}
