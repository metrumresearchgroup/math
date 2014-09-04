#include <stan/math/matrix/trace_inv_quad_form_ldlt.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, trace_inv_quad_form_ldlt) {
  stan::math::matrix_d A(4,4), B(4,2);
  stan::math::LDLT_factor<double,-1,-1> ldlt_A;
  A << 
    9.0,  3.0, 3.0,   3.0, 
    3.0, 10.0, 2.0,   2.0,
    3.0,  2.0, 7.0,   1.0,
    3.0,  2.0, 1.0, 112.0;
  B << 
    100, 10,
    0,  1,
    -3, -3,
    5,  2;
  ldlt_A.compute(A);
  ASSERT_TRUE(ldlt_A.success());

  EXPECT_FLOAT_EQ(1439.1061766207,
                  trace_inv_quad_form_ldlt(ldlt_A,B));
  EXPECT_FLOAT_EQ((B.transpose() * A.inverse() * B).trace(),
                  trace_inv_quad_form_ldlt(ldlt_A,B));
}

TEST(MathMatrix, trace_inv_quad_form_ldlt_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,3);
  m1 << 1, nan, 1.1,
        nan, 4.1, 4.5,
        nan, nan, 5.1;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 10.1, 100, 2,
        1.0, 1, nan,
        -10, -12, 6;

  Eigen::MatrixXd m3(3,3);
  m3 << 1, nan, 5.1,
        3, 4.1, 1.1,
        nan, 6, nan;
        
  Eigen::MatrixXd m4(3,3);
  m4 << 10.1, 100, 4,
        0.1, nan, 5,
        -10, -12, 7;
        
  Eigen::MatrixXd m1c(3,3);
  m1c << 1, 1.1, 4,
        1.3, 4.1, 3,
        4.1, 1.6, 1;
        
  Eigen::MatrixXd m2c(3,3);
  m2c << 10.1, 100, 2,
        1.0, 1, 7,
        -10, -12, 6;

  Eigen::MatrixXd m3c(3,3);
  m3c << 1, 0.1, 5.1,
        3, 4.1, 1.1,
        4, 6, .4;
        
  Eigen::MatrixXd m4c(3,3);
  m4c << 10.1, 100, 1,
        0.1, 6, 6,
        -10, -12, 1.1;

  stan::math::LDLT_factor<double,-1,-1> ldlt_m1;
  ldlt_m1.compute(m1);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m2;
  ldlt_m2.compute(m2);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m3;
  ldlt_m3.compute(m3);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m4;
  ldlt_m4.compute(m4);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m1c;
  ldlt_m1c.compute(m1c);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m2c;
  ldlt_m2c.compute(m2c);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m3c;
  ldlt_m3c.compute(m3c);
  stan::math::LDLT_factor<double,-1,-1> ldlt_m4c;
  ldlt_m4c.compute(m4c);

  using stan::math::trace_inv_quad_form_ldlt;
  using boost::math::isnan;
   
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m1, m1));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m1, m2));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m1, m3));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m1, m4));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m2, m1));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m2, m2));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m2, m3));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m2, m4));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m3, m1));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m3, m2));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m3, m3));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m3, m4));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m4, m1));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m4, m2));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m4, m3));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m4, m4));
   
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m1c, m1));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m1c, m2));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m1c, m3));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m1c, m4));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m2c, m1));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m2c, m2));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m2c, m3));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m2c, m4));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m3c, m1));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m3c, m2));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m3c, m3));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m3c, m4));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m4c, m1));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m4c, m2));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m4c, m3));
  EXPECT_PRED1(isnan<double>, trace_inv_quad_form_ldlt(ldlt_m4c, m4));
}
