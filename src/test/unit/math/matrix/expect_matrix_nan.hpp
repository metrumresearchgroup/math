#ifndef TEST_MATH_MATRIX_EXPECT_MATRIX_NAN_HPP
#define TEST_MATH_MATRIX_EXPECT_MATRIX_NAN_HPP

#include <gtest/gtest.h>
#include <stan/math/matrix/Eigen.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

template <int R, int C>
void expect_matrix_nan(const Eigen::Matrix<double, R, C> & mat) {
  for (int i = 0, size_ = mat.size(); i < size_; i++)
    EXPECT_PRED1(boost::math::isnan<double>, mat(i));
}

#endif
