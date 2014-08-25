#ifndef TEST_MATH_MATRIX_EXPECT_MATRIX_NAN_HPP
#define TEST_MATH_MATRIX_EXPECT_MATRIX_NAN_HPP

#include <gtest/gtest.h>
#include <stan/math/matrix/Eigen.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

template <int R, int C>
void expect_matrix_is_nan(const Eigen::Matrix<double, R, C> & mat) {
  for (int i = 0, size_ = mat.size(); i < size_; i++)
    EXPECT_PRED1(boost::math::isnan<double>, mat(i));
}

void expect_matrix_is_nan(const std::vector<double> & vec) {
  for (size_t i = 0, size_ = vec.size(); i < size_; i++)
    EXPECT_PRED1(boost::math::isnan<double>, vec[i]);
}

template <int R, int C>
void expect_matrix_not_nan(const Eigen::Matrix<double, R, C> & mat) {
  for (int i = 0, size_ = mat.size(); i < size_; i++)
    EXPECT_FALSE(boost::math::isnan<double>(mat(i)));
}

void expect_matrix_not_nan(const std::vector<double> & vec) {
  for (size_t i = 0, size_ = vec.size(); i < size_; i++)
    EXPECT_FALSE(boost::math::isnan<double>(vec[i]));
}

#endif
