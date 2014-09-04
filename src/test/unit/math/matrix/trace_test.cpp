#include <stan/math/matrix/trace.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, trace) {
  using stan::math::trace;
  stan::math::matrix_d m;
  EXPECT_FLOAT_EQ(0.0,trace(m));
  m = stan::math::matrix_d(1,1);
  m << 2.3;
  EXPECT_FLOAT_EQ(2.3,trace(m));
  m = stan::math::matrix_d(2,3);
  m << 1, 2, 3, 4, 5, 6;
  EXPECT_FLOAT_EQ(6.0,trace(m));
}

TEST(MathMatrix, trace_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,2);
  m1 << 1, nan,
        nan, 4.1,
        nan, nan;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 10.1, 100, 2,
        1.0, 1, nan,
        -10, -12, 6;

  Eigen::MatrixXd m3(3,3);
  m3 << 1, nan, 5.1,
        3, 4.1, 1.1,
        nan, 6, nan;
        
  Eigen::MatrixXd m4(3,2);
  m4 << 10.1, 100,
        0.1, nan,
        -10, -12;

  using stan::math::trace;
  using boost::math::isnan;
   
  EXPECT_DOUBLE_EQ(5.1, trace(m1));
  EXPECT_DOUBLE_EQ(17.1, trace(m2));
  EXPECT_PRED1(isnan<double>, trace(m3));
  EXPECT_PRED1(isnan<double>, trace(m4));
}
