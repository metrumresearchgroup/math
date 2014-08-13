#include <stan/math/matrix/log_determinant.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix, log_determinant) {

  using stan::math::log_determinant;

  Eigen::MatrixXd m1(3,3);
  m1 << 14,     1,   3,
        3.1, 1.2, 4.1,
        3.4, 4.1, 1.1; //negative determinant
        
  Eigen::MatrixXd m2(3,3);
  m2 << 14,     1,   3,
        3.1, 13.2, 4.1,
        3.4, 4.1, 11.1; //positive determinant

  using stan::math::log_determinant;
        
  EXPECT_DOUBLE_EQ(log_determinant(m1),
    log(std::abs(14*1.2*1.1 + 4.1*3.4 + 3.1*4.1*3 -
        3.4*1.2*3 - 4.1*4.1*14 - 1.1*3.1)));
  EXPECT_DOUBLE_EQ(log_determinant(m2),
    log(14*13.2*11.1 + 4.1*3.4 + 3.1*4.1*3 -
        3.4*13.2*3 - 4.1*4.1*14 - 11.1*3.1));

}


TEST(MathMatrix, log_determinant_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,3);
  m1 << 100, 1, 3,
        nan, 100, 4.1,
        3.1, 4, 100;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 10.1, 1.0, 1.0,
        1.1, 5.4, 1.5,
        1.3, 1.5, nan;
        
  Eigen::MatrixXd m3(3,3);
  m3 << 10.1, 1.6, nan,
        1.1, 15.4, 1.5,
        1.3, 13.5, 1.7;
        
  Eigen::MatrixXd m4(3,3);
  m4 << 0, 1.0, 1.0,
        0, 0.1, 1.5,
        0, 1.5, nan;
        
  using stan::math::log_determinant;
  using boost::math::isnan;
    
  EXPECT_PRED1(isnan<double>, log_determinant(m1));
  EXPECT_PRED1(isnan<double>, log_determinant(m2));
  EXPECT_PRED1(isnan<double>, log_determinant(m3));
  EXPECT_PRED1(isnan<double>, log_determinant(m4));
}
