cmake_minimum_required(VERSION 2.8.3)
project(trajectory_estimation)

find_package(catkin REQUIRED)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        rospy
        std_msgs
        tf
        message_generation
        image_transport
        cv_bridge
        image_geometry
        laser_geometry
        sensor_msgs
        pcl_ros
        pcl_conversions
        nav_msgs
)
find_package(OpenCV REQUIRED)

catkin_package(
  INCLUDE_DIRS include
#  LIBRARIES libsurf
#  CATKIN_DEPENDS other_catkin_pkg
#  DEPENDS system_lib
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive -std=c++11")

include_directories(include ${catkin_INCLUDE_DIRS})

#add_executable(feature_detection_node src/feature_detection.cpp)
#target_link_libraries(feature_detection_node ${OpenCV_LIBS} ${catkin_LIBRARIES} ${PROJECT_SOURCE_DIR}/include/libsurf.so)

#add_executable(vo_node src/visodo.cpp)
#target_link_libraries(vo_node ${OpenCV_LIBS} ${catkin_LIBRARIES} ${PROJECT_SOURCE_DIR}/include/libsurf.so)

add_executable(${PROJECT_NAME}_node src/visual_odometry.cpp)
target_link_libraries(${PROJECT_NAME}_node ${OpenCV_LIBS} ${catkin_LIBRARIES} ${PROJECT_SOURCE_DIR}/include/libsurf.so)
