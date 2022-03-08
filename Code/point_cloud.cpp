#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;
using namespace cv;

#define K_MATRIX_DIM 3
#define SYSTEM_VARS 4

pcl::visualization::PCLVisualizer::Ptr simpleVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<pcl::PointXYZRGB> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

int main() {
    Mat rgb_image = imread("Dataset/00000-color.png");
    Mat d_image = imread("Dataset/00000-depth.png");

    /*namedWindow("RGB Image", WINDOW_AUTOSIZE );
    imshow("RGB Image", rgb_image);
    namedWindow("Depth Image", WINDOW_AUTOSIZE );
    imshow("Depth Image", d_image);*/

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>); 

    // my camera parameters
    float cx = 330.2f; //optical center x coordinate
    float cy = 254.4f; //optical center y coordinate
    float fx = 522.3f; //focal length x
    float fy = 523.4f; //focal length x

    pcl::PointXYZRGB point;

    int imageDepth;
    cv::Vec3b pixel;
    int i = 0;
    for (int imageWidth = 0; imageWidth < rgb_image.cols; imageWidth++) {
        for (int imageHeight = 0; imageHeight < rgb_image.rows; imageHeight++) {
            imageDepth = d_image.at<int>(imageWidth, imageHeight);
            point.x = (imageWidth - cx) / fx;
            point.y = (imageHeight - cy) / fy;
            point.z = imageDepth / sqrt(1 + pow(point.x, 2)+ pow(point.y, 2));
            point.x *= point.z;
            point.y *= point.z;
            pixel = rgb_image.at<cv::Vec3b>(imageWidth, imageHeight);
            point.r = pixel[2];
            point.g = pixel[1];
            point.b = pixel[0];
            cloud->push_back(point);
        }
    }

    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = simpleVis(cloud);

    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
    }
    
    waitKey(0);

	return 0;
}