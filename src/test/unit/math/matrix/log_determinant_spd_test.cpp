#include <stan/math/matrix/log_determinant_spd.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix, log_determinant_spd_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m0(3,3);
  m0 << 10, 1, 3.2,
        1.1, 10, 4.1,
        0.1, 4.1, 10;

  Eigen::MatrixXd m1(3,3);
  m1 << 10, 1, 3.2,
        nan, 10, 4.1,
        0.1, 4.1, 10;

        
  Eigen::MatrixXd m2(3,3);
  m2 << 10, 1, 3.2,
        1.1, nan, 4.1,
        0.1, 4.1, 10;
          
  Eigen::MatrixXd m3(3,3);
  m3 << 10, 1, nan,
        1.1, 10, 4.1,
        0.1, 4.1, 10;
        
  using stan::math::log_determinant_spd;
  using boost::math::isnan;
    
  EXPECT_FLOAT_EQ(6.7100382, log_determinant_spd(m0));
  EXPECT_PRED1(isnan<double>, log_determinant_spd(m1));
  EXPECT_PRED1(isnan<double>, log_determinant_spd(m2));
  EXPECT_FLOAT_EQ(6.7100382, log_determinant_spd(m3));
}
