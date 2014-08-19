#include <stan/math/matrix/quad_form_diag.hpp>
#include <test/unit/math/matrix/expect_matrix_eq.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::quad_form_diag;

TEST(MathMatrix,quadFormDiag) {

  Matrix<double,Dynamic,Dynamic> m(1,1);
  m << 3;

  Matrix<double,Dynamic,1> v(1);
  v << 9;

  Matrix<double,Dynamic,Dynamic> v_m(1,1);
  v_m << 9;
  
  expect_matrix_eq(v_m * m * v_m, quad_form_diag(m,v));
}
TEST(MathMatrix,quadFormDiag2) {
  Matrix<double,Dynamic,Dynamic> m(3,3);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;

  Matrix<double,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;
  
  expect_matrix_eq(v_m * m * v_m, quad_form_diag(m,v));

  Matrix<double,1,Dynamic> rv(3);
  rv << 1, 2, 3;
  expect_matrix_eq(v_m * m * v_m, quad_form_diag(m,rv));
}

TEST(MathMatrix,quadFormDiagException) {
  Matrix<double,Dynamic,Dynamic> m(2,2);
  m << 
    2, 3,
    4, 5;
  EXPECT_THROW(quad_form_diag(m,m), std::domain_error);

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;
  EXPECT_THROW(quad_form_diag(m,v), std::domain_error);
  
  Matrix<double,Dynamic,Dynamic> m2(3,2);
  m2 << 
    2, 3,
    4, 5,
    6, 7;
    
  Matrix<double,Dynamic,1> v2(2);
  v2 << 1, 2;

  EXPECT_THROW(quad_form_diag(m2,v), std::domain_error);
  EXPECT_THROW(quad_form_diag(m2,v2), std::domain_error);
}


TEST(MathMatrix,quadFormDiag2_dv_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  Matrix<double,Dynamic,Dynamic> m(3,3);
  Matrix<double,1,Dynamic> rv(3);
  Matrix<double,Dynamic,1> v(3);

  m << nan, 2, 3, 4, 5, 6, 7, 8, 9;
  v << nan, 2, 3;
  rv << nan, 2, 3;

  Matrix<double, Dynamic, Dynamic> res1 = quad_form_diag(m, v);
  Matrix<double, Dynamic, Dynamic> res2 = quad_form_diag(m, rv);

  EXPECT_FLOAT_EQ(20, res1(1,1));
  EXPECT_FLOAT_EQ(36, res1(1,2));
  EXPECT_FLOAT_EQ(48, res1(2,1));
  EXPECT_FLOAT_EQ(81, res1(2,2));
  EXPECT_TRUE(boost::math::isnan(res1(0,0)));
  EXPECT_TRUE(boost::math::isnan(res1(0,1)));
  EXPECT_TRUE(boost::math::isnan(res1(0,2)));
  EXPECT_TRUE(boost::math::isnan(res1(1,0)));
  EXPECT_TRUE(boost::math::isnan(res1(2,0)));

  EXPECT_FLOAT_EQ(20, res2(1,1));
  EXPECT_FLOAT_EQ(36, res2(1,2));
  EXPECT_FLOAT_EQ(48, res2(2,1));
  EXPECT_FLOAT_EQ(81, res2(2,2));
  EXPECT_TRUE(boost::math::isnan(res2(0,0)));
  EXPECT_TRUE(boost::math::isnan(res2(0,1)));
  EXPECT_TRUE(boost::math::isnan(res2(0,2)));
  EXPECT_TRUE(boost::math::isnan(res2(1,0)));
  EXPECT_TRUE(boost::math::isnan(res2(2,0)));
  EXPECT_TRUE(boost::math::isnan(res2(2,0)));

}
