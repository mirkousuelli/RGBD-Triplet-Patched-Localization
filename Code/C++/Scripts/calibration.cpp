#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define K_MATRIX_DIM 3
#define SYSTEM_VARS 4

void getUpperCholesky(Mat* input, Mat* output) {
	/**
	 * Cholesky factorization takes as input a matrix and returns
	 * as output the upper triangularization decomposition matrix.
	 */

	assert(input->rows == input->cols);
	assert(output->rows == output->cols);
	assert(input->rows == output->cols);

	int n = input->rows;

	for (int i = 0; i < n; ++i) {
		for (int k = 0; k < i; ++k) {
			double value = input->at<double>(i, k);
			for (int j = 0; j < k; ++j)
				value -= output->at<double>(i, j) * output->at<double>(k, j);
			output->at<double>(i, k) = value / output->at<double>(k, k);
		}
		double value = input->at<double>(i, i);
		for (size_t j = 0; j < i; ++j)
			value -= output->at<double>(i, j) * output->at<double>(i, j);
		output->at<double>(i, i) = sqrt(value);
	}
	output->at<double>(n - 1, n - 1) = 1.0;
	*output = output->t();
}

void cleanCalibrationMatrix(Mat* K) {
	/**
	 * This function clean the calibration matrix after that the
	 * upper cholesky triangular matrix has been inverted because the
	 * component at (2,2) is equal to 0.9999999999999999, while others are
	 * set as -0.0 instead of 0.0
	 */

	assert(K->rows == 3 && K->cols == 3);

	K->at<double>(0, 1) = 0.0;
	K->at<double>(1, 0) = 0.0;
	K->at<double>(2, 2) = 1.0;
}

void getCalibrationMatrix(Point3d v1, Point3d v2, Point3d v3, Mat* K) {
	/**
	 * Gets the image of the absolute conic from a rectified face, two
	 * horizontal vanishing points related to the rectified face, a vanishing
	 * point orthogonal to the other two and the homography reconstruction.
	 * 
	 * v1 : vertical vanishing point
	 * v2 : 1st horizontal vanishing point
	 * v3 : 2nd horizontal vanishing point
	 */

	// left-hand matrix (coefficients)
	double a[] = {	
		// v1.T * W * v2 = 0
		(v1.x * v2.x + v1.y * v2.y), (v1.z * v2.x + v1.x * v2.z), (v1.z * v2.y + v1.y * v2.z),

		// v1.T * W * v3 = 0
		(v1.x * v3.x + v1.y * v3.y), (v1.z * v3.x + v1.x * v3.z), (v1.z * v3.y + v1.y * v3.z),

		// v3.T * W * v2 = 0 
		(v3.x * v2.x + v3.y * v2.y), (v3.z * v2.x + v3.x * v2.z), (v3.z * v2.y + v3.y * v2.z), 
	};
	
	// right-hand vector
	double b[] = {
		-(v1.z * v2.z), 
		-(v1.z * v3.z),
		-(v3.z * v2.z) 
	};

	// linear system : A * X = B
	Mat A(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, a);
	Mat B(K_MATRIX_DIM, 1, CV_64F, b);
	Mat X(K_MATRIX_DIM, 1, CV_64F);
	solve(A, B, X);

	/* Image of Absolute Conic.
	 * Since we have 3 vanishing points, namely 3 constraints, and 4 variables, the solution we are
	 * looking for is:
	 * w = [ a  0  b
	 *       0  a  c
	 *       b  c  d ]
	 * I set the variable 'd' as free, hence I assume it as 1 value:
	 * w = [ a  0  b
	 *       0  a  c
	 *       b  c  1 ]
	 */
	double w[] = {
		X.at<double>(0,0), 0, X.at<double>(1,0),
		0, X.at<double>(0,0), X.at<double>(2,0),
		X.at<double>(1,0), X.at<double>(2,0), 1
	};
	Mat W(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, w);

	// Cholesky upper decomposition
	double l[] = {
		0, 0, 0,
		0, 0, 0,
		0, 0, 0
	};
	Mat L(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, l);
	getUpperCholesky(&W, &L);

	// Calibration matrix: K = inv(chol(W))
	invert(L, *K);

	// needed for some flaws due to cv::invert()
	cleanCalibrationMatrix(K);
}

void normalizeElement(Point3d* p) {
	p->x /= p->z;
	p->y /= p->z;
	p->z = 1;
}

int main() {
	// 1st vanishing point (vertical)
	Point3d v1_up_a(77, 334, 1);
	Point3d v1_up_b(54, 208, 1);
	Point3d l1_up = v1_up_a.cross(v1_up_b);
	normalizeElement(&l1_up);
	Point3d v1_down_a(567, 379, 1);
	Point3d v1_down_b(97, 241, 1);
	Point3d l1_down = v1_down_a.cross(v1_down_b);
	normalizeElement(&l1_down);
	Point3d v1 = l1_up.cross(l1_down);
	normalizeElement(&v1);

	// 2nd vanishing point (horizontal)
	Point3d v2_up_a(248, 391, 1);
	Point3d v2_up_b(55, 209, 1);
	Point3d l2_up = v2_up_a.cross(v2_up_b);
	normalizeElement(&l2_up);
	Point3d v2_down_a(173, 474, 1);
	Point3d v2_down_b(78, 337, 1);
	Point3d l2_down = v2_down_a.cross(v2_down_b);
	normalizeElement(&l2_down);
	Point3d v2 = l2_up.cross(l2_down);
	normalizeElement(&v2);

	// 3rd vanishing point (horizontal)
	Point3d v3_up_a(248, 391, 1);
	Point3d v3_up_b(597, 240, 1);
	Point3d l3_up = v3_up_a.cross(v3_up_b);
	normalizeElement(&l3_up);
	Point3d v3_down_a(458, 450, 1);
	Point3d v3_down_b(567, 378, 1);
	Point3d l3_down = v3_down_a.cross(v3_down_b);
	normalizeElement(&l3_down);
	Point3d v3 = l3_up.cross(l3_down);
	normalizeElement(&v3);

	// calibration matrix K
	Mat K(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F);
	getCalibrationMatrix(v1, v2, v3, &K);
	cout << "K = " << endl << " " << K << endl << endl;

	return 0;
}