#include <stan/math/matrix/LDLT_factor.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, LDLT_factor_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m0(3,3);
  m0 << 100, 1, 3.2,
        1.1, 100, 4.1,
        3.1, 4.1, 100;
  stan::math::LDLT_factor<double,-1,-1> ldlt_m0;
  ldlt_m0.compute(m0);
  ASSERT_TRUE(ldlt_m0.success());

  Eigen::MatrixXd m1(3,3);
  m1 << 100, 1, 3.2,
        nan, 100, 4.1,
        3.1, 4.1, 100;
  stan::math::LDLT_factor<double,-1,-1> ldlt_m1;
  ldlt_m1.compute(m1);
  ASSERT_FALSE(ldlt_m1.success());

        
  Eigen::MatrixXd m2(3,3);
  m2 << 100, 1, 3.2,
        1.1, nan, 4.1,
        3.1, 4.1, 100;
  stan::math::LDLT_factor<double,-1,-1> ldlt_m2;
  ldlt_m2.compute(m2);
  ASSERT_FALSE(ldlt_m2.success());
          
  Eigen::MatrixXd m3(3,3);
  m3 << 100, 1, nan,
        1.1, 100, 4.1,
        3.1, 4.1, 100;
  stan::math::LDLT_factor<double,-1,-1> ldlt_m3;
  ldlt_m3.compute(m3);
  ASSERT_TRUE(ldlt_m3.success());
}
