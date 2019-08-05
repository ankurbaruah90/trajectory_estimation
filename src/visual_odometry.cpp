#include "visual_odometry/vo_features.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

using namespace cv;
using namespace std;

#define MAX_FRAME 250
#define MIN_NUM_FEAT 1500
#define SCALE 1

ros::NodeHandle *n_;
Mat R_f, t_f;
Mat prevImage;
vector<Point2f> prevFeatures;
Mat traj = Mat::zeros(600, 600, CV_8UC3);
double focal = 940.9240116;
cv::Point2d pp(636.445312, 359.391479);

void imageCallBack(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat currImage;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv::cvtColor(cv_ptr->image, currImage, CV_BGR2GRAY);

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
        E = findEssentialMat(currFeatures, prevFeatures, focal, pp, RANSAC, 0.999, 1.0, mask);
        recoverPose(E, currFeatures, prevFeatures, R, t, focal, pp, mask);

        if(t_f.empty())
            t_f = t.clone();

        if(R_f.empty())
            R_f = R.clone();

        if((SCALE > 0.1) && (t.at<double>(2) > t.at<double>(0)) && (t.at<double>(2) > t.at<double>(1)))
        {
            t_f = t_f + SCALE*(R_f*t);
            R_f = R*R_f;
        }

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

    cout << "\nRotation " << R_f << "\nTranslation " << t_f << endl;

    int x = int(t_f.at<double>(0)) + traj.cols/2;
    int y = int(t_f.at<double>(2)) + traj.rows/2;
    circle(traj, Point(x, y) ,1, CV_RGB(255,0,0), 2);
    cv::imshow( "Trajectory", traj );
    cv::waitKey(1);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "feature_detection_node");
    n_ = new ros::NodeHandle;

    ros::Rate rate(20);
    ros::Subscriber image_subscriber = n_->subscribe("/camera_front/rgb/image_raw", 1, imageCallBack);
//    ros::Subscriber image_subscriber = n_->subscribe("/camera/color/image_raw", 1, imageCallBack);

    while(ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }

    delete n_;
    return 0;
}

