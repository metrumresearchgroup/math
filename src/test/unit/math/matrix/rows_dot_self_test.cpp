#include <stan/math/matrix/rows_dot_self.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix,rows_dot_self) {
  using stan::math::rows_dot_self;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,rows_dot_self(m1)(0,0),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x,y;
  x = rows_dot_self(m2);
  EXPECT_NEAR(13.0,x(0,0),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  y = rows_dot_self(m3);
  EXPECT_NEAR(13.0,y(0,0),1E-12);
  EXPECT_NEAR(41.0,y(1,0),1E-12);
}

TEST(MathMatrix,rows_dot_self_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3, 2);
  m1 << 14, nan,
        3, 4.1,
        10, 4.1;

  Eigen::VectorXd vr;
  
  using stan::math::rows_dot_self;
  using boost::math::isnan;

  vr = rows_dot_self(m1);
  EXPECT_PRED1(isnan<double>, vr(0));
  EXPECT_DOUBLE_EQ(3*3 + 4.1*4.1, vr(1));
  EXPECT_DOUBLE_EQ(10*10 + 4.1*4.1, vr(2));
}
