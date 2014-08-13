#include <stan/math/matrix/log_determinant_ldlt.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <stan/math/matrix/determinant.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix, log_determinant_ldlt) {
  using stan::math::determinant;
  using std::log;
  using std::fabs;
  
  stan::math::matrix_d x(2,2);  
  stan::math::LDLT_factor<double,-1,-1> ldlt_x;

  x << 2, 1, 1, 3;
  ldlt_x.compute(x);
  ASSERT_TRUE(ldlt_x.success());
  
  EXPECT_FLOAT_EQ(log(fabs(determinant(x))),
                  stan::math::log_determinant_ldlt(ldlt_x));

  x << 1, 0, 0, 3;
  ldlt_x.compute(x);
  ASSERT_TRUE(ldlt_x.success());
  EXPECT_FLOAT_EQ(log(3.0),
                  stan::math::log_determinant_ldlt(ldlt_x));
}

TEST(MathMatrix, log_determinant_ldlt_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m0(3,3);
  m0 << 10, 1, 3.2,
        1.1, 10, 4.1,
        0.1, 4.1, 10;
  stan::math::LDLT_factor<double,-1,-1> ldlt_m0;
  ldlt_m0.compute(m0);
  ASSERT_TRUE(ldlt_m0.success());

  Eigen::MatrixXd m1(3,3);
  m1 << 10, 1, 3.2,
        nan, 10, 4.1,
        0.1, 4.1, 10;
  stan::math::LDLT_factor<double,-1,-1> ldlt_m1;
  ldlt_m1.compute(m1);
  ASSERT_FALSE(ldlt_m1.success());

        
  Eigen::MatrixXd m2(3,3);
  m2 << 10, 1, 3.2,
        1.1, nan, 4.1,
        0.1, 4.1, 10;
  stan::math::LDLT_factor<double,-1,-1> ldlt_m2;
  ldlt_m2.compute(m2);
  ASSERT_FALSE(ldlt_m2.success());
          
  Eigen::MatrixXd m3(3,3);
  m3 << 10, 1, nan,
        1.1, 10, 4.1,
        0.1, 4.1, 10;
  stan::math::LDLT_factor<double,-1,-1> ldlt_m3;
  ldlt_m3.compute(m3);
  ASSERT_TRUE(ldlt_m3.success());
        
  using stan::math::log_determinant_ldlt;
  using boost::math::isnan;
    
  EXPECT_FLOAT_EQ(6.7100382, log_determinant_ldlt(ldlt_m0));
  EXPECT_PRED1(isnan<double>, log_determinant_ldlt(ldlt_m1));
  EXPECT_PRED1(isnan<double>, log_determinant_ldlt(ldlt_m2));
  EXPECT_FLOAT_EQ(6.7100382, log_determinant_ldlt(ldlt_m3));
}
