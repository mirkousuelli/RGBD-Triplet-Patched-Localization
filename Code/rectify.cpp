#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void normalize(Point3d* p) {
    p->x /= p->z;
    p->y /= p->z;
    p->z = 1;
}

Mat getAffineMatrix() {
    /* points */
    Point3d p1_a(247, 392, 1);
    Point3d p1_b(55, 208, 1);
    Point3d p2_a(595, 240, 1);
    Point3d p2_b(335, 159, 1);

    /* affine transformation */
    Point3d line1 = p1_a.cross(p1_b);
    normalize(&line1);

    Point3d line2 = p2_a.cross(p2_b);
    normalize(&line2);

    Point3d vp1 = line1.cross(line2);
    normalize(&vp1);

    Point3d line3 = p1_a.cross(p2_a);
    normalize(&line3);
    
    Point3d line4 = p1_b.cross(p2_b);
    normalize(&line4);

    Point3d vp2 = line3.cross(line4);
    normalize(&vp2);

    Point3d inf_line = vp1.cross(vp2);
    normalize(&inf_line);

    double affine_matrix[] = {1, 0, 0, 0, 1, 0, inf_line.x, inf_line.y, inf_line.z};
    Mat H_aff(3, 3, CV_64F, affine_matrix);

    std::cout << "H_aff = " << std::endl << " " << H_aff.t() << std::endl << std::endl;

    return H_aff.t();
}

Mat getEuclideanMatrix() {
    /* points */
    Point3d p1_a(247, 392, 1);
    Point3d p1_b(55, 208, 1);
    Point3d p2_a(595, 240, 1);
    Point3d p2_b(335, 159, 1);

    /* euclidean transformation */
    Point3d line1 = p1_a.cross(p2_b);
    normalize(&line1);

    Point3d line2 = p2_a.cross(p1_b);
    normalize(&line2);

    Point3d vp1 = line1.cross(line2);
    normalize(&vp1);

    Point3d line3 = p1_a.cross(p2_a);
    normalize(&line3);
    
    Point3d line4 = p1_b.cross(p2_b);
    normalize(&line4);

    Point3d vp2 = line3.cross(line4);
    normalize(&vp2);

    Point3d inf_line = vp1.cross(vp2);
    normalize(&inf_line);

    double affine_matrix[] = {1, 0, 0, 0, 1, 0, inf_line.x, inf_line.y, inf_line.z};
    Mat H_aff(3, 3, CV_64F, affine_matrix);

    std::cout << "H_aff = " << std::endl << " " << H_aff.t() << std::endl << std::endl;

    return H_aff.t();
}

int main(int argc, char** argv)
{
    // Read image from file 
    Mat img = imread("Dataset/00000-color.png");;
    Mat aff;

    //if fail to read the image
    if ( img.empty() ) { 
        cout << "Error loading the image" << endl;
        return -1; 
    }

    //Create a window
    namedWindow("Image", 1);

    //show the image
    imshow("Image", img);

    Mat aff_matrix = getAffineMatrix();
    Size size(img.cols, img.rows);
    warpPerspective(img, aff, aff_matrix, size);

    //Create a window
    namedWindow("Affine Image", 2);

    //show the image
    imshow("Affine Image", aff);

    // Wait until user press some key
    waitKey(0);

    return 0;
}