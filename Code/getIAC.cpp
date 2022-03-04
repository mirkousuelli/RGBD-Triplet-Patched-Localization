#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

#define K_MATRIX_DIM 3
#define SYSTEM_VARS 4

Mat getCalibrationMatrix(Point3_<double> vp_vert, Point3_<double> vp_ort_1, Point3_<double> vp_ort_2, Mat H) {
	/**
	 * Gets the image of the absolute conic from a rectified face, two
	 * horizontal vanishing points related to the rectified face, a vanishing
	 * point orthogonal to the other two and the homography reconstruction.
	 */

	// destination matrix in order to invert the homography H
	Mat invH(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F);

	// invert homography H into invH
	invert(H, invH);

	// get the first and second column of invH
	Point3_<double> h1(invH.col(0));
	Point3_<double> h2(invH.col(1));

	std::cout << "*vertical vp = " << std::endl << " " << vp_vert << std::endl << std::endl;
	std::cout << "*1st horizontal vp = " << std::endl << " " << vp_ort_1 << std::endl << std::endl;
	std::cout << "*2nd horizontal vp = " << std::endl << " " << vp_ort_2 << std::endl << std::endl;
	std::cout << "*homography = " << std::endl << " " << H << std::endl << std::endl;
	std::cout << "*inv_homography = " << std::endl << " " << invH << std::endl << std::endl;
	std::cout << "*h1  = " << std::endl << " " << h1 << std::endl << std::endl;
	std::cout << "*h2  = " << std::endl << " " << h2 << std::endl << std::endl;

	/* Now we set a linear system as: A * x = b
	 * Where A gets the vanishing points coefficients according to the absolute conic constraints
	 * x are the 4 unknowns a, b, c, d and b is the right-hand component
	 */

	// left-hand matrix (coefficients)
	double a[] = {	
			// Equation 1 : a*v(1)*v1(1) + b*v(1)*v1(3) + v(2)*v1(2) + c*v(2)*v1(3) + b*v(3)*v1(1) + c*v(3)*v1(2) + d*v(3)*v1(3);
			vp_vert.x * vp_ort_1.x, vp_vert.x * vp_ort_1.z + vp_vert.z * vp_ort_1.x, vp_vert.y * vp_ort_1.z + vp_vert.z * vp_ort_1.y, vp_vert.z * vp_ort_1.z,

			// Equation 2 : a*v(1)*v1(1) + b*v(1)*v1(3) + v(2)*v1(2) + c*v(2)*v1(3) + b*v(3)*v1(1) + c*v(3)*v1(2) + d*v(3)*v1(3);
			vp_vert.x * vp_ort_2.x, vp_vert.x * vp_ort_2.z + vp_vert.z * vp_ort_2.x, vp_vert.y * vp_ort_2.z + vp_vert.z * vp_ort_2.y, vp_vert.z * vp_ort_2.z,

			// Equation 3 : a*h2(1)*h1(1) + b*h2(1)*h1(3) + h2(2)*h1(2) + c*h2(2)*h1(3) + b*h2(3)*h1(1) + c*h2(3)*h1(2) + d*h2(3)*h1(3);
			h2.x * h1.x, (h2.x * h1.z + h2.z * h1.x), (h2.y * h1.z + h2.z * h1.y), h2.z * h1.z,

			// Equation 4 : a*h1(1)*h1(1) + b*h1(1)*h1(3) + h1(2)*h1(2) + c*h1(2)*h1(3) + b*h1(3)*h1(1) + c*h1(3)*h1(2) + d*h1(3)*h1(3) - (a*h2(1)*h2(1) + b*h2(1)*h2(3) + h2(2)*h2(2) + c*h2(2)*h2(3) + b*h2(3)*h2(1) + c*h2(3)*h2(2) + d*h2(3)*h2(3));
			h1.x * h1.x - h2.x * h2.x, 2 * (h1.x * h1.z - h2.x * h2.z), 2 * (h1.y * h1.z - h2.y * h2.z), h1.z * h1.z - h2.z * h2.z 
	};
	Mat A(SYSTEM_VARS, SYSTEM_VARS, CV_64F, a);

	// right-hand vector
	double b[] = {	
		- vp_vert.y * vp_ort_1.y, 
		- vp_vert.y * vp_ort_2.y,
		- h2.y * h1.y,
		- h1.y * h1.y + h2.y * h2.y 
	};
	Mat B(SYSTEM_VARS, 1, CV_64F, b);

	// unknown variables: [a, b, c, d]
	double x[] = { 1, 1, 1, 1};
	Mat X(SYSTEM_VARS, 1, CV_64F, x);

	std::cout << "A = " << std::endl << " " << A << std::endl << std::endl;
	std::cout << "b = " << std::endl << " " << B << std::endl << std::endl;

	// solve the linear system as : x = inv(A) * b
	solve(A, B, X);

	std::cout << "x = " << std::endl << " " << X << std::endl << std::endl;

	/* The reconstruction assumes null skew factor, thus:
	 * iac	=	[ a 0 b ]
	 *			[ 0 1 c ]
	 *			[ b c d ]
	 * Remembering that a = 1 / aspect_ratio.
	 */
	double iac[] = { 	
		X.at<double>(0,0), 0, X.at<double>(1,0),
		0, 1, X.at<double>(2,0),
		X.at<double>(1,0), X.at<double>(2,0), X.at<double>(3,0) 
	};
	Mat IAC(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, iac);
	std::cout << "IAC = " << std::endl << " " << IAC << std::endl << std::endl;

	double alfa = sqrt(iac[0]);
	double u0 = -iac[2] / pow(alfa, 2);
	double v0 = -iac[5];
	double fy = sqrt(iac[8] - pow(alfa, 2) * pow(u0, 2) - pow(v0, 2));
	double fx = fy / alfa;
	double k[] = {
		fx, 0, u0,
		0, fy, v0,
		0, 0, 1
	};
	Mat K = Mat(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, k);

	std::cout << "K = " << std::endl << " " << K << std::endl << std::endl;

	return Mat(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, k);
}

int main() {
	/**
	 * Vertical vanishing points:
	 * 1) 1.0e+03 * 0.4939  -1.8320   0.0010
	 * 
	 * Horizontal vanishing points:
	 * 1) 1.0e+04 * 1.5043   0.1081   0.0001
	 * 2) 1.0e+03 * 0.4231   1.1908   0.0010
	 */

	Point3_<double> vp(493.9, -1832.0, 1.0);
	Point3_<double> vh1(15043.0, 1081.0, 1.0);
	Point3_<double> vh2(423.1, 1190.8, 1.0);

	// affine transformation matrix
	double h_aff[] = {1, 0, -0.0001, 0, 1, -0.0008, 0, 0, 1};
	Mat H_aff = Mat(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, h_aff);

	// euclidean transformation matrix
	double h_met[] = {1.2398, -0.3136, 0, -0.3136, 1.4399, 0, 0, 0, 1};
	Mat H_met = Mat(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, h_met);

	Mat homography = H_met * H_aff;

	std::cout << "vertical vp = " << std::endl << " " << vp << std::endl << std::endl;
	std::cout << "1st horizontal vp = " << std::endl << " " << vh1 << std::endl << std::endl;
	std::cout << "2nd horizontal vp = " << std::endl << " " << vh2 << std::endl << std::endl;
	std::cout << "affine = " << std::endl << " " << H_aff << std::endl << std::endl;
	std::cout << "euclidean = " << std::endl << " " << H_met << std::endl << std::endl;
	std::cout << "homography = " << std::endl << " " << homography << std::endl << std::endl;

	Mat K = getCalibrationMatrix(vp, vh1, vh2, homography);

	//std::cout << "K = " << std::endl << " " << K << std::endl << std::endl;

	return 0;
}