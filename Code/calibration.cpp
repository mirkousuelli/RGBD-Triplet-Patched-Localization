#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
//#include <bits/stdc++.h>

using namespace std;
using namespace cv;

#define K_MATRIX_DIM 3
#define SYSTEM_VARS 4

template <typename scalar_type> class matrix {
public:
    matrix(size_t rows, size_t columns)
        : rows_(rows), columns_(columns), elements_(rows * columns) {}
 
    matrix(size_t rows, size_t columns, scalar_type value)
        : rows_(rows), columns_(columns), elements_(rows * columns, value) {}
 
    matrix(size_t rows, size_t columns,
        const std::initializer_list<std::initializer_list<scalar_type>>& values)
        : rows_(rows), columns_(columns), elements_(rows * columns) {
        assert(values.size() <= rows_);
        size_t i = 0;
        for (const auto& row : values) {
            assert(row.size() <= columns_);
            std::copy(begin(row), end(row), &elements_[i]);
            i += columns_;
        }
    }
 
    size_t rows() const { return rows_; }
    size_t columns() const { return columns_; }
 
    const scalar_type& operator()(size_t row, size_t column) const {
        assert(row < rows_);
        assert(column < columns_);
        return elements_[row * columns_ + column];
    }
    scalar_type& operator()(size_t row, size_t column) {
        assert(row < rows_);
        assert(column < columns_);
        return elements_[row * columns_ + column];
    }
private:
    size_t rows_;
    size_t columns_;
    std::vector<scalar_type> elements_;
};
 
template <typename scalar_type>
void print(std::ostream& out, const matrix<scalar_type>& a) {
    size_t rows = a.rows(), columns = a.columns();
    out << std::fixed << std::setprecision(5);
    for (size_t row = 0; row < rows; ++row) {
        for (size_t column = 0; column < columns; ++column) {
            if (column > 0)
                out << ' ';
            out << std::setw(9) << a(row, column);
        }
        out << '\n';
    }
}
 
template <typename scalar_type>
matrix<scalar_type> cholesky_factor(const matrix<scalar_type>& input) {
    assert(input.rows() == input.columns());
    size_t n = input.rows();
    matrix<scalar_type> result(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t k = 0; k < i; ++k) {
            scalar_type value = input(i, k);
            for (size_t j = 0; j < k; ++j)
                value -= result(i, j) * result(k, j);
            result(i, k) = value/result(k, k);
        }
        scalar_type value = input(i, i);
        for (size_t j = 0; j < i; ++j)
            value -= result(i, j) * result(i, j);
        result(i, i) = std::sqrt(value);
    }
    return result;
}
 
void print_cholesky_factor(const matrix<double>& matrix) {
    std::cout << "Matrix:\n";
    print(std::cout, matrix);
    std::cout << "Cholesky factor:\n";
    print(std::cout, cholesky_factor(matrix));
}

void getCalibrationMatrix(Point3d v1, Point3d v2, Point3d v3) {
	/**
	 * Gets the image of the absolute conic from a rectified face, two
	 * horizontal vanishing points related to the rectified face, a vanishing
	 * point orthogonal to the other two and the homography reconstruction.
     * 
     * v1 : vertical vanishing point
     * v2 : 1st horizontal vanishing point
     * v3 : 2nd horizontal vanishing point
	 */

	std::cout << "* vertical vp = " << std::endl << " " << v1 << std::endl << std::endl;
	std::cout << "* 1st horizontal vp = " << std::endl << " " << v2 << std::endl << std::endl;
	std::cout << "* 2nd horizontal vp = " << std::endl << " " << v3 << std::endl << std::endl;

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

    Mat A(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, a);
    Mat B(K_MATRIX_DIM, 1, CV_64F, b);
    Mat X(K_MATRIX_DIM, 1, CV_64F);

    solve(A, B, X);

    std::cout << "A = " << std::endl << " " << A << std::endl << std::endl;
    std::cout << "B = " << std::endl << " " << B << std::endl << std::endl;
    std::cout << "X = " << std::endl << " " << X << std::endl << std::endl;

    matrix<double> matrix2(3, 3,
       { { X.at<double>(0,0), 0, X.at<double>(1,0) },
        { 0, X.at<double>(0,0), X.at<double>(2,0) },
        { X.at<double>(1,0), X.at<double>(2,0), 1 } });
    print_cholesky_factor(matrix2);
 
    double l[] = {0.00195, 0.00000, -0.13076,
     0.00000, 0.00195, -2.00738,
     0,  0, 1 };
    Mat L(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, l);

	/* The reconstruction assumes null skew factor, thus:
	 * iac	=	[ a 0 b ]
	 *			[ 0 1 c ]
	 *			[ b c d ]
	 * Remembering that a = 1 / aspect_ratio.
	 */
	/*double iac[] = { 	
		X.at<double>(0,0), 0, X.at<double>(1,0),
		0, X.at<double>(0,0), X.at<double>(2,0),
		X.at<double>(1,0), X.at<double>(2,0), 1
	};
	Mat IAC(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, iac);
	std::cout << "IAC = " << std::endl << " " << IAC << std::endl << std::endl;*/

    Mat K;
    invert(L, K);

	/*double alfa = sqrt(iac[0]);
	double u0 = -iac[2] / pow(alfa, 2);
	double v0 = -iac[5];
	double fy = sqrt(iac[8] - pow(alfa, 2) * pow(u0, 2) - pow(v0, 2));
	double fx = fy / alfa;
	double k[] = {
		fx, 0, u0,
		0, fy, v0,
		0, 0, 1
	};
	Mat K = Mat(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, k);*/
    cout << endl;
	std::cout << "K = " << std::endl << " " << K << std::endl << std::endl;
    cout << "PORCODDDUE E' ANDATA! Marco, metto il codice a posto domani se stai leggendo questo eseguibile... non se capisce un cazzo ma so stanco, so le 2.30 am" << endl << endl;


	//return //Mat(K_MATRIX_DIM, K_MATRIX_DIM, CV_64F, k);
}

void normalize(Point3d* p) {
    p->x /= p->z;
    p->y /= p->z;
    p->z = 1;
}

int main() {
    Point3d v1_up_a(77, 334, 1);
    Point3d v1_up_b(54, 208, 1);
    Point3d l1_up = v1_up_a.cross(v1_up_b);
    normalize(&l1_up);
    Point3d v1_down_a(567, 379, 1);
    Point3d v1_down_b(97, 241, 1);
    Point3d l1_down = v1_down_a.cross(v1_down_b);
    normalize(&l1_down);

    Point3d v2_up_a(248, 391, 1);
    Point3d v2_up_b(55, 209, 1);
    Point3d l2_up = v2_up_a.cross(v2_up_b);
    normalize(&l2_up);
    Point3d v2_down_a(173, 474, 1);
    Point3d v2_down_b(78, 337, 1);
    Point3d l2_down = v2_down_a.cross(v2_down_b);
    normalize(&l2_down);

    Point3d v3_up_a(248, 391, 1);
    Point3d v3_up_b(597, 240, 1);
    Point3d l3_up = v3_up_a.cross(v3_up_b);
    normalize(&l3_up);
    Point3d v3_down_a(458, 450, 1);
    Point3d v3_down_b(567, 378, 1);
    Point3d l3_down = v3_down_a.cross(v3_down_b);
    normalize(&l3_down);

	Point3d v1 = l1_up.cross(l1_down);
    normalize(&v1);
	Point3d v2 = l2_up.cross(l2_down);
    normalize(&v2);
	Point3d v3 = l3_up.cross(l3_down);
    normalize(&v3);

	getCalibrationMatrix(v1, v2, v3);

	//std::cout << "K = " << std::endl << " " << K << std::endl << std::endl;

	return 0;
}