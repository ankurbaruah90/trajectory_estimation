#include <iostream>
#include <math.h>
#include <opencv/cv.h>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Geometry>
#include <Eigen/src/Geometry/EulerAngles.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include "OpenSURFcpp/src/surflib.h"
#include "opencv2/features2d/features2d.hpp"

#define USE_ORB 1
#define RANSAC_THRESHOLD 10
#define MAX_INLIER_DIST 2
#define WINDOW 1

using namespace std;

int image_count = 0;
ros::NodeHandle *n_;
cv::Mat rmat, tmat;
deque <cv::Mat> src_img, src_desc;
deque <vector <cv::KeyPoint> > src_kp;
double origin_x, origin_y;
deque <cv::Point2f> drift_window;
cv::Ptr<cv::DescriptorMatcher> matcher_;

void cross_check_matching(cv::Ptr<cv::DescriptorMatcher>& descriptorMatcher,
                         const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                         vector<cv::DMatch>& filteredMatches12, int knn=10)
{
    filteredMatches12.clear();
    assert(!descriptors1.empty());
    vector<vector<cv::DMatch> > matches12, matches21;

    descriptorMatcher->knnMatch(descriptors1, descriptors2, matches12, knn);
    descriptorMatcher->knnMatch(descriptors2, descriptors1, matches21, knn);

    for(size_t m = 0; m < matches12.size(); m++)
    {
        bool findCrossCheck = false;
        for(size_t fk = 0; fk < matches12[m].size(); fk++)
        {
            cv::DMatch forward = matches12[m][fk];
            for(size_t bk = 0; bk < matches21[forward.trainIdx].size(); bk++)
            {
                cv::DMatch backward = matches21[forward.trainIdx][bk];
                if( backward.trainIdx == forward.queryIdx )
                {
                    filteredMatches12.push_back(forward);
                    findCrossCheck = true;
                    break;
                }
            }
            if(findCrossCheck)
                break;
        }
    }
}

void match_keypoints(std::vector<cv::KeyPoint> new_kp,
                     cv::Mat new_desc, cv::Mat new_img)
{
    if(src_img.size() > WINDOW)
    {
        src_img.pop_front();
        src_desc.pop_front();
        src_kp.pop_front();
    }
    src_img.push_back(new_img);
    src_desc.push_back(new_desc);
    src_kp.push_back(new_kp);

    for(int frame = 0; frame < src_img.size() - 1; frame++)
    {
        if((src_desc[frame].rows < 15) || (new_desc.rows < 15))
        {
            ROS_WARN("Too few descriptors found for image matching");
            return;
        }

        std::vector<cv::DMatch> matches;
        cross_check_matching(matcher_, new_desc, src_desc[frame], matches, 2);

        if(matches.size() < 5)
        {
            ROS_WARN("Too few matches found for estimating homography");
            return;
        }

        vector<int> queryIdxs,  trainIdxs;
        for(size_t i = 0; i < matches.size(); i++)
        {
            queryIdxs.push_back(matches[i].queryIdx);
            trainIdxs.push_back(matches[i].trainIdx);
        }

        cv::Mat err;
        vector<uchar> status;
        cv::Size winSize = cv::Size(81,81);
        vector<cv::Point2f> new_points, src_points, perspective_transformed_points;
        cv::KeyPoint::convert(src_kp[frame], src_points, trainIdxs);
        cv::KeyPoint::convert(new_kp, new_points, queryIdxs);
        cv::calcOpticalFlowPyrLK(src_img[frame], new_img, src_points, perspective_transformed_points, status, err, winSize);

        vector<cv::Point2f> points1filtered, points2filtered;
        for(size_t i1 = 0; i1 < status.size(); i1++)
        {
            if(status[i1])
            {
//                cout << "Frame " << frame << " --- " << src_points[i1].x << "x" << src_points[i1].y << " --- " << perspective_transformed_points[i1].x << "x" << perspective_transformed_points[i1].y << " --- " << norm(new_points[i1] - perspective_transformed_points[i1]) << endl;
                if((norm(new_points[i1] - perspective_transformed_points[i1]) <= int(MAX_INLIER_DIST)))
                {
                    points1filtered.push_back(new_points[i1]);
                    points2filtered.push_back(src_points[i1]);
                    cv::circle(new_img, points1filtered.back(), 2, cv::Scalar(0, 0, 255), 1);
                    cv::circle(new_img, points2filtered.back(), 2, cv::Scalar(255, 0, 0), 1);
                    cv::arrowedLine(new_img, points2filtered.back(), points1filtered.back(), cv::Scalar(0, 255, 0), 2, 8, 0, 0.2);
                }
            }
        }

        cv::Mat transformation_matrix, rvec, tvec;
        if(points2filtered.size() > 5 && points1filtered.size() > 5)
        {
            transformation_matrix = cv::findEssentialMat(cv::Mat(points2filtered), cv::Mat(points1filtered));
            cv::recoverPose(transformation_matrix, cv::Mat(points2filtered), cv::Mat(points1filtered), rvec, tvec);

            if(rmat.empty() || tmat.empty())
            {
                rmat = rvec.clone();
                tmat = tvec.clone();
            }
            else
            {
                rmat = rvec * rmat;
                tmat = tmat - (rmat*tvec);
            }

            cout << tmat.at<double>(0) << "," << tmat.at<double>(1) << "," << tmat.at<double>(2) << endl;
//            cout << "Rotation Matrix\n" << rmat << endl;
//            cout << "Translation Matrix\n" << tmat << endl;
//            origin_x += transformation_matrix.at<double>(0,2);
//            origin_y += transformation_matrix.at<double>(1,2);

        }
    }
    cv::imshow("matching", new_img);
    cv::waitKey(1);
}

//void match_keypoints(std::vector<cv::KeyPoint> new_kp,
//                     cv::Mat new_desc, cv::Mat new_img)
//{
//    if(!new_desc.empty())
//    {
//        if(src_img.empty())
//            src_img = new_img;

//        if((src_desc.rows < 15) || (new_desc.rows < 15))
//        {
//            ROS_WARN("Too few descriptors found for image matching");
//            return;
//        }

//        std::vector<cv::DMatch> matches;
//        cross_check_matching(matcher_, new_desc, src_desc, matches, 2);

//        if(matches.size() < 5)
//        {
//            ROS_WARN("Too few matches found for estimating homography");
//            return;
//        }

//        vector<int> queryIdxs,  trainIdxs;
//        for(size_t i = 0; i < matches.size(); i++)
//        {
//            queryIdxs.push_back(matches[i].queryIdx);
//            trainIdxs.push_back(matches[i].trainIdx);
//        }

//        cv::Mat err;
//        vector<uchar> status;
//        cv::Size winSize = cv::Size(81,81);
//        vector<cv::Point2f> points1, points2, perspective_transformed_points;
//        cv::KeyPoint::convert(src_kp, points2, trainIdxs);
//        cv::KeyPoint::convert(new_kp, points1, queryIdxs);
//        cv::calcOpticalFlowPyrLK(new_img, src_img, points2, perspective_transformed_points, status, err, winSize);

//        vector<cv::Point2f> points1filtered, points2filtered;
//        for(size_t i1 = 0; i1 < status.size(); i1++)
//        {
//            if(status[i1])
//            {
//                if((norm(points1[i1] - perspective_transformed_points[i1]) <= int(MAX_INLIER_DIST)))
//                {
//                    points1filtered.push_back(points1[i1]);
//                    points2filtered.push_back(points2[i1]);
//                    cv::circle(new_img, points1filtered.back(), 2, cv::Scalar(0, 0, 255), 1);
//                    cv::circle(new_img, points2filtered.back(), 2, cv::Scalar(255, 0, 0), 1);
//                    cv::arrowedLine(new_img, points1filtered.back(), points2filtered.back(), cv::Scalar(0, 255, 0), 2, 8, 0, 0.2);
//                }
//            }
//        }

//        if(points2filtered.size() > 5 && points1filtered.size() > 5)
//        {
//            cv::Mat transformation_matrix = cv::findEssentialMat(cv::Mat(points2filtered), cv::Mat(points1filtered));
//            origin_x += transformation_matrix.at<double>(0,2);
//            origin_y += transformation_matrix.at<double>(1,2);
//            cout << "Transformation\n" << transformation_matrix << endl;
//            cout << origin_x << "," << origin_y << endl;
//        }

//        double flow_divergence;
//        for(size_t sp = 0; sp < points1filtered.size(); sp++)
//        {
//            for(size_t dp = sp+1; dp < status.size(); dp++)
//            {
//                if(status[sp] && status[dp])
//                {
//                    double dx_src = points1filtered[sp].x - points1filtered[dp].x;
//                    double dy_src = points1filtered[sp].y - points1filtered[dp].y;
//                    double eq_dist_src = sqrt(pow(dx_src, 2) + pow(dy_src, 2));             //previous

//                    double dx_dst = points2filtered[sp].x - points2filtered[dp].x;
//                    double dy_dst = points2filtered[sp].y - points2filtered[dp].y;
//                    double eq_dist_dst = sqrt(pow(dx_dst, 2) + pow(dy_dst, 2));             //current

//                    flow_divergence += ((eq_dist_src > 0) ? (eq_dist_src - eq_dist_dst)/eq_dist_src : 0);
//                }
//            }

//        }

//        double current_time = ros::Time::now().toSec();
//        double delta_time = current_time - previous_time;
//        flow_divergence = flow_divergence/(status.size() * delta_time);
//        previous_time = current_time;
//        cout << "Divergence " << flow_divergence << " time " << delta_time << endl;

//        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 528.995513, 0, 303.036604, 0, 527.983687, 257.009383, 0, 0, 1);
//        cv::Mat distortionCoefficients = (cv::Mat_<double>(1, 5) << 0.131585, -0.191955, -0.005917, -0.003070, 0.000000);

//        cv::Mat rvec, tvec, rmat;
//        cv::solvePnP(cv::Mat(points1filtered), cv::Mat(points2filtered), cameraMatrix, distortionCoefficients, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);

//        cv::Mat transformation_matrix = cv::findHomography(cv::Mat(points2filtered), cv::Mat(points1filtered), CV_RANSAC, RANSAC_THRESHOLD);

//        if(drift_window.size() > 5)
//            drift_window.pop_front();
//        drift_window.push_back(cv::Point2f(transformation_matrix.at<double>(0,2),transformation_matrix.at<double>(1,2)));

//        double x_pos = 0, y_pos = 0;
//        for(int it = 0; it != drift_window.size(); it++)
//        {
//            x_pos += drift_window[it].x;
//            y_pos += drift_window[it].y;
//        }

//        x_pos = x_pos/drift_window.size();
//        y_pos = y_pos/drift_window.size();
//        origin_x += x_pos;
//        origin_y += y_pos;


//        cout << "Translation\n" << tvec << endl;
//        cout << "Rotation\n" << rvec << endl;
//        cout << "Transformation\n" << transformation_matrix << endl;

//        cv::imshow("matching", new_img);
//    }
//    else
//        cv::imshow("matching", new_img);
//    cv::waitKey(1);
//}

void imageCallBack(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    cv::Mat binary, binary_tmp;
    try
    {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

//    char filename[200];
//    sprintf(filename, "/home/ankur/Desktop/VOimage/%d.png", image_count++);
//    cv::imwrite(filename, cv_ptr->image);

    ///gaussian pre-filtering and sharpening---------------------------------------------
    cv::cvtColor(cv_ptr->image, binary, CV_BGR2GRAY);
    cv::GaussianBlur(binary, binary_tmp, cv::Size(1,1), cv::BORDER_DEFAULT);
    cv::addWeighted(binary_tmp, 1.5, binary, -0.5, 0, binary);

    ///Extracting features---------------------------------------------------------------
    cv::Mat map_desc;
    std::vector<cv::KeyPoint> map_kp;
#if USE_ORB
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create(1000, 1.2f, 4, 31, 0, 2, cv::ORB::HARRIS_SCORE, 24, 20);
    detector->detect(binary, map_kp, cv::noArray());
    detector->compute(binary, map_kp, map_desc);
#else
    int count = 0;
    IplImage *img = new IplImage(binary);
    std::vector<Ipoint> opensurf_keyp;
    surfDetDes(img, opensurf_keyp, true, 8, 100, INIT_SAMPLE, 0.0);
    map_kp.resize(opensurf_keyp.size());
    map_desc = cv::Mat(cv::Size(64,opensurf_keyp.size()), CV_32FC1, 0.0);
    for(std::vector <Ipoint>::iterator it = opensurf_keyp.begin(); it != opensurf_keyp.end(); ++it){
        map_kp[count].pt.x = (*it).x;
        map_kp[count].pt.y = (*it).y;
        map_kp[count].angle = (*it).orientation;
        cv::Mat tmp = cv::Mat(cv::Size(64,1), CV_32FC1, (*it).descriptor);
        for(int i = 0; i < tmp.cols; i++)
            map_desc.at<float>(count, i) = tmp.at<float>(0, i);
        count++;
    }
    delete img;
#endif

    match_keypoints(map_kp, map_desc, cv_ptr->image);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "feature_detection_node");
    n_ = new ros::NodeHandle;

    ros::Rate rate(30);
    ros::Subscriber image_subscriber = n_->subscribe("/camera_front/rgb/image_raw", 1, imageCallBack);
//    ros::Subscriber image_subscriber = n_->subscribe("/camera/rgb/image_raw", 1, imageCallBack);
//    ros::Subscriber image_subscriber = n_->subscribe("/camera/color/image_raw", 1, imageCallBack);

    matcher_ = cv::DescriptorMatcher::create("BruteForce");
    while(ros::ok())
    {
        ros::spinOnce();
        rate.sleep();
    }

    delete n_;
    return 0;
}
