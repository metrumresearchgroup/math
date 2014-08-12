#include <stan/math/matrix/determinant.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,dimensionValidation) {
  using stan::math::determinant;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  Matrix<double,Dynamic,Dynamic> x(3,3);
  x << 1, 2, 3, 1, 4, 9, 1, 8, 27;

  ASSERT_FALSE(boost::math::isnan(determinant(x)));

  Matrix<double,Dynamic,Dynamic> xx(3,2);
  xx << 1, 2, 3, 1, 4, 9;
  EXPECT_THROW(stan::math::determinant(xx),std::domain_error);
}

TEST(MathMatrix, determinant_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,3);
  m1 << 14, 1, 3,
        nan, 10, 4.1,
        3.1, 4.1, 8;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 10.1, 1.0, 1.0,
        1.1, 5.4, 1.5,
        1.3, 1.5, nan;
        
  Eigen::MatrixXd m3(3,3);
  m3 << 0, 1.0, 1.0,
        0, 0.1, 1.5,
        0, 1.5, nan;
        
  using stan::math::determinant;
  using boost::math::isnan;
    
  EXPECT_PRED1(isnan<double>, determinant(m1));
  EXPECT_PRED1(isnan<double>, determinant(m2));
  EXPECT_PRED1(isnan<double>, determinant(m3));
}
