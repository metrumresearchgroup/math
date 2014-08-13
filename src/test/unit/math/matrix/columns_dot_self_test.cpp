#include <stan/math/matrix/columns_dot_self.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix,columns_dot_self) {
  using stan::math::columns_dot_self;

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,columns_dot_self(m1)(0,0),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m2(1,2);
  m2 << 2.0, 3.0;
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> x;
  x = columns_dot_self(m2);
  EXPECT_NEAR(4.0,x(0,0),1E-12);
  EXPECT_NEAR(9.0,x(0,1),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m3(2,2);
  m3 << 2.0, 3.0, 4.0, 5.0;
  x = columns_dot_self(m3);
  EXPECT_NEAR(20.0,x(0,0),1E-12);
  EXPECT_NEAR(34.0,x(0,1),1E-12);
}

TEST(MathMatrix,columns_dot_self_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(2,3);
  m1 << 14, nan, 3,
        nan, 10, 4.1;
  
  Eigen::RowVectorXd vr;
  
  using stan::math::columns_dot_self;
  using boost::math::isnan;

  vr = columns_dot_self(m1);
  EXPECT_PRED1(isnan<double>, vr(0));
  EXPECT_PRED1(isnan<double>, vr(1));
  EXPECT_DOUBLE_EQ(3*3 + 4.1*4.1, vr(2));
}
