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

void expect_matrix_eq_or_both_nan(
  const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> & mat1,
  const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> & mat2) {
  EXPECT_EQ(mat1.rows(), mat2.rows());
  EXPECT_EQ(mat1.cols(), mat2.cols());
  for (int i = 0, size_ = mat1.size(); i < size_; i++) {
    if (boost::math::isnan<double>(mat1(i)))
      EXPECT_PRED1(boost::math::isnan<double>, mat2(i));
    else
      EXPECT_FLOAT_EQ(mat1(i), mat2(i));
  }
}

void expect_matrix_eq_or_both_nan(const std::vector<double> & vec1,
                                  const std::vector<double> & vec2) {
  EXPECT_EQ(vec1.size(), vec2.size());
  for (size_t i = 0, size_ = vec1.size(); i < size_; i++) {
    if (boost::math::isnan<double>(vec1[i]))
      EXPECT_PRED1(boost::math::isnan<double>, vec2[i]);
    else
      EXPECT_FLOAT_EQ(vec1[i], vec2[i]);
  }
}

#endif
