#include <stan/math/matrix/trace_quad_form.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, trace_quad_form_mat) {
  using stan::math::trace_quad_form;
  using stan::math::matrix_d;
  
  matrix_d ad(4,4);
  matrix_d bd(4,2);
  double res;
  bd << 100, 10,
          0,  1,
         -3, -3,
          5,  2;
  ad << 2.0,  3.0, 4.0,   5.0, 
  6.0, 10.0, 2.0,   2.0,
  7.0,  2.0, 7.0,   1.0,
  8.0,  2.0, 1.0, 112.0;
  
  // double-double
  res = trace_quad_form(ad,bd);
  EXPECT_FLOAT_EQ(26758, res);
}

TEST(MathMatrix, trace_quad_form_nan) {
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

  using stan::math::trace_quad_form;
  using boost::math::isnan;
   
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1, m1));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1, m2));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1, m3));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1, m4));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2, m1));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2, m2));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2, m3));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2, m4));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3, m1));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3, m2));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3, m3));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3, m4));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4, m1));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4, m2));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4, m3));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4, m4));
   
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1c, m1));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1c, m2));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1c, m3));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1c, m4));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2c, m1));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2c, m2));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2c, m3));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2c, m4));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3c, m1));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3c, m2));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3c, m3));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3c, m4));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4c, m1));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4c, m2));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4c, m3));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4c, m4));
   
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1, m1c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1, m2c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1, m3c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m1, m4c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2, m1c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2, m2c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2, m3c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m2, m4c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3, m1c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3, m2c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3, m3c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m3, m4c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4, m1c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4, m2c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4, m3c));
  EXPECT_PRED1(isnan<double>, trace_quad_form(m4, m4c));
}
