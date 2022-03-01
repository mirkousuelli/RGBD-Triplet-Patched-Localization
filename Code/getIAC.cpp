#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace cv;

#define K_MATRIX_DIM 3
#define SYSTEM_VARS 4

Mat getIAC(InputArray vp_vert, InputArray vp_ort_1, InputArray vp_ort_2, Mat H) {
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
	Mat h1 = Mat(1, K_MATRIX_DIM, CV_64F, invH.col(0));
	Mat h2 = Mat(1, K_MATRIX_DIM, CV_64F, invH.col(1));

	/* Now we set a linear system as: A * x = b
	 * Where A gets the vanishing points coefficients according to the absolute conic constraints
	 * x are the 4 unknowns a, b, c, d and b is the right-hand component
	 */

	// left-hand matrix (coefficients)
	Mat A = Mat(SYSTEM_VARS, SYSTEM_VARS, CV_64F, 
		{	
			// Equation 1 : a*v(1)*v1(1) + b*v(1)*v1(3) + v(2)*v1(2) + c*v(2)*v1(3) + b*v(3)*v1(1) + c*v(3)*v1(2) + d*v(3)*v1(3);
			vp_vert.at<double>(0,0) * vp_ort_1.at<double>(0,0), vp_vert.at<double>(0,0) * vp_ort_1.at<double>(2,0) + vp_vert.at<double>(2,0) * vp_ort_1.at<double>(0,0), vp_vert.at<double>(1,0) * vp_ort_1.at<double>(2,0) + vp_vert.at<double>(2,0) * vp_ort_1.at<double>(1,0), vp_vert.at<double>(2,0) * vp_ort_1.at<double>(2,0),

			// Equation 2 : a*v(1)*v1(1) + b*v(1)*v1(3) + v(2)*v1(2) + c*v(2)*v1(3) + b*v(3)*v1(1) + c*v(3)*v1(2) + d*v(3)*v1(3);
			vp_vert.at<double>(0,0) * vp_ort_2.at<double>(0,0), vp_vert.at<double>(0,0) * vp_ort_2.at<double>(2,0) + vp_vert.at<double>(2,0) * vp_ort_2.at<double>(0,0), vp_vert.at<double>(1,0) * vp_ort_2.at<double>(2,0) + vp_vert.at<double>(2,0) * vp_ort_2.at<double>(1,0), vp_vert.at<double>(2,0) * vp_ort_2.at<double>(2,0),

			// Equation 3 : a*h2(1)*h1(1) + b*h2(1)*h1(3) + h2(2)*h1(2) + c*h2(2)*h1(3) + b*h2(3)*h1(1) + c*h2(3)*h1(2) + d*h2(3)*h1(3);
			h2.at<double>(0,0) * h1.at<double>(0,0), (h2.at<double>(0,0) * h1.at<double>(2,0) + h2.at<double>(2,0) * h1.at<double>(0,0)), (h2.at<double>(1,0) * h1.at<double>(2,0) + h2.at<double>(2,0) * h1.at<double>(1,0)), h2.at<double>(2,0) * h1.at<double>(2,0),

			// Equation 4 : a*h1(1)*h1(1) + b*h1(1)*h1(3) + h1(2)*h1(2) + c*h1(2)*h1(3) + b*h1(3)*h1(1) + c*h1(3)*h1(2) + d*h1(3)*h1(3) - (a*h2(1)*h2(1) + b*h2(1)*h2(3) + h2(2)*h2(2) + c*h2(2)*h2(3) + b*h2(3)*h2(1) + c*h2(3)*h2(2) + d*h2(3)*h2(3));
			h1.at<double>(0,0) * h1.at<double>(0,0) - h2.at<double>(0,0) * h2.at<double>(0,0), 2 * (h1.at<double>(0,0) * h1.at<double>(2,0) - h2.at<double>(0,0) * h2.at<double>(2,0)), 2 * (h1.at<double>(1,0) * h1.at<double>(2,0) - h2.at<double>(1,0) * h2.at<double>(2,0)), h1.at<double>(2,0) * h1.at<double>(2,0) - h2.at<double>(2,0) * h2.at<double>(2,0) 
		}
	);

	// right-hand vector
	Mat b = Mat(SYSTEM_VARS, 1, CV_64F, 
		{	
			- vp_vert.at<double>(1,0) * vp_ort_1.at<double>(1,0), 
			- vp_vert.at<double>(1,0) * vp_ort_2.at<double>(1,0),
			- h2.at<double>(1,0) * h1.at<double>(1,0),
			- h1.at<double>(1,0) * h1.at<double>(1,0) + h2.at<double>(1,0) * h2.at<double>(1,0) 
		}
	);

	// unknown variables: [a, b, c, d]
	Mat x(SYSTEM_VARS, 1, CV_64F);

	// solve the linear system as : x = inv(A) * b
	solve(A, b, x)

	/* The reconstruction assumes null skew factor, thus:
	 * iac	=	[ a 0 b ]
	 *			[ 0 1 c ]
	 *			[ b c d ]
	 * Remembering that a = 1 / aspect_ratio.
	 */
	return Mat(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, 
		{ 	
			x.at<double>(0,0), 0, x.at<double>(1,0),
			0, 1, x.at<double>(2,0),
			x.at<double>(1,0), x.at<double>(2,0), x.at<double>(3,0) 
		}
	);
}