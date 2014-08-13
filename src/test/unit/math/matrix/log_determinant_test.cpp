#include <stan/math/matrix/log_determinant.hpp>
#include <gtest/gtest.h>

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
