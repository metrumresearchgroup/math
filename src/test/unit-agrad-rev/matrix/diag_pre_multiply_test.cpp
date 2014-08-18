#include <vector>
#include <stan/agrad.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <test/unit-agrad-rev/matrix/expect_matrix_eq.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;
using stan::agrad::var;
using stan::math::diag_pre_multiply;

TEST(MathMatrix,diagPreMultiply2_vv) {
  Matrix<var,Dynamic,Dynamic> m(3,3);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;
  
  Matrix<var,Dynamic,Dynamic> v_m_times_m = v_m * m;

  expect_matrix_eq(v_m_times_m, diag_pre_multiply(v,m));

  Matrix<var,1,Dynamic> rv(3);
  rv << 1, 2, 3;

  expect_matrix_eq(v_m_times_m, diag_pre_multiply(rv,m));
}

TEST(MathMatrix,diagPreMultiply2_vd) {
  Matrix<var,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;
  
  Matrix<var,Dynamic,Dynamic> v_m_times_m1 = v_m * m1;

  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  expect_matrix_eq(v_m_times_m1, diag_pre_multiply(v,m2));
}

TEST(MathMatrix,diagPreMultiply2_dv) {
  Matrix<double,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;


  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;

  Matrix<var,Dynamic,Dynamic> v_m_times_m2 = v_m * m2;

  expect_matrix_eq(v_m_times_m2, diag_pre_multiply(v,m1));

}


TEST(MathMatrix,diagPreMultiplyGrad_vv) {
  Matrix<var,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;

  std::vector<var> xs1;
  for (int i = 0; i < 9; ++i)
    xs1.push_back(m1(i));
  for (int i = 0; i < 3; ++i)
    xs1.push_back(v(i));

  Matrix<var,Dynamic,Dynamic> v_pre_multipy_m1 = diag_pre_multiply(v, m1);

  var norm1 = v_pre_multipy_m1.norm();

  std::vector<double> g1;
  norm1.grad(xs1,g1);


  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;

  std::vector<var> xs2;
  for (int i = 0; i < 9; ++i)
    xs2.push_back(m2(i));
  for (int i = 0; i < 3; ++i)
    xs2.push_back(v_m(i,i));

  
  Matrix<var,Dynamic,Dynamic> v_m_times_m = v_m * m2;

  var norm2 = v_m_times_m.norm();
  std::vector<double> g2;
  norm2.grad(xs2,g2);
  
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i], g2[i]);
}

TEST(MathMatrix,diagPreMultiplyGrad_vd) {
  Matrix<var,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;

  std::vector<var> xs1;
  for (int i = 0; i < 9; ++i)
    xs1.push_back(m1(i));

  Matrix<var,Dynamic,Dynamic> v_pre_multipy_m1 = diag_pre_multiply(v, m1);

  var norm1 = v_pre_multipy_m1.norm();

  std::vector<double> g1;
  norm1.grad(xs1,g1);


  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);  // OK to use var, just for comparison
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;

  std::vector<var> xs2;
  for (int i = 0; i < 9; ++i)
    xs2.push_back(m2(i));
  
  Matrix<var,Dynamic,Dynamic> v_m_times_m = v_m * m2;

  var norm2 = v_m_times_m.norm();
  std::vector<double> g2;
  norm2.grad(xs2,g2);
  
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i], g2[i]);
}

TEST(MathMatrix,diagPreMultiplyGrad_dv) {
  Matrix<double,Dynamic,Dynamic> m1(3,3);
  m1 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;

  std::vector<var> xs1;
  for (int i = 0; i < 3; ++i)
    xs1.push_back(v(i));

  Matrix<var,Dynamic,Dynamic> v_pre_multipy_m1 = diag_pre_multiply(v, m1);

  var norm1 = v_pre_multipy_m1.norm();

  std::vector<double> g1;
  norm1.grad(xs1,g1);


  Matrix<var,Dynamic,Dynamic> m2(3,3);
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;

  Matrix<var,Dynamic,Dynamic> v_m(3,3);  // OK to use var, just for comparison
  v_m << 
    1, 0, 0,
    0, 2, 0,
    0, 0, 3;

  std::vector<var> xs2;
  for (int i = 0; i < 3; ++i)
    xs2.push_back(v_m(i,i));
  
  Matrix<var,Dynamic,Dynamic> v_m_times_m = v_m * m2;

  var norm2 = v_m_times_m.norm();
  std::vector<double> g2;
  norm2.grad(xs2,g2);
  
  EXPECT_EQ(g1.size(), g2.size());
  for (size_t i = 0; i < g1.size(); ++i)
    EXPECT_FLOAT_EQ(g1[i], g2[i]);
}


TEST(MathMatrix,diagPreMultiplyException) {
  Matrix<var,Dynamic,Dynamic> m(2,2);
  m << 
    2, 3,
    4, 5;
  EXPECT_THROW(diag_pre_multiply(m,m), std::domain_error);

  Matrix<var,Dynamic,1> v(3);
  v << 1, 2, 3;
  EXPECT_THROW(diag_pre_multiply(v,m), std::domain_error);
}

TEST(MathMatrix,diagPreMultiply2_vv_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  Matrix<var,Dynamic,Dynamic> m(3,3);
  Matrix<var,Dynamic,1> v(3);
  Matrix<var,1,Dynamic> rv(3);

  m << nan, 2, 3, 4, nan, 6, 7, 8, 9;
  v << 1, nan, 3;
  rv << 1, nan, 3;

  Matrix<var,Dynamic,Dynamic> res1 = diag_pre_multiply(v,m);
  Matrix<var,Dynamic,Dynamic> res2 = diag_pre_multiply(rv,m);

  EXPECT_TRUE(boost::math::isnan(res1(0,0).val()));
  EXPECT_TRUE(boost::math::isnan(res1(1,0).val()));
  EXPECT_TRUE(boost::math::isnan(res1(1,1).val()));
  EXPECT_TRUE(boost::math::isnan(res1(1,2).val()));
  EXPECT_FLOAT_EQ(2, res1(0,1).val());
  EXPECT_FLOAT_EQ(3, res1(0,2).val());
  EXPECT_FLOAT_EQ(21, res1(2,0).val());
  EXPECT_FLOAT_EQ(24, res1(2,1).val());
  EXPECT_FLOAT_EQ(27, res1(2,2).val());

  EXPECT_TRUE(boost::math::isnan(res2(0,0).val()));
  EXPECT_TRUE(boost::math::isnan(res2(1,0).val()));
  EXPECT_TRUE(boost::math::isnan(res2(1,1).val()));
  EXPECT_TRUE(boost::math::isnan(res2(1,2).val()));
  EXPECT_FLOAT_EQ(2, res2(0,1).val());
  EXPECT_FLOAT_EQ(3, res2(0,2).val());
  EXPECT_FLOAT_EQ(21, res2(2,0).val());
  EXPECT_FLOAT_EQ(24, res2(2,1).val());
  EXPECT_FLOAT_EQ(27, res2(2,2).val());
}

TEST(MathMatrix,diagPreMultiply2_vd_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  Matrix<var,Dynamic,Dynamic> m(3,3);
  Matrix<double,Dynamic,1> v(3);
  Matrix<double,1,Dynamic> rv(3);

  m << nan, 2, 3, 4, nan, 6, 7, 8, 9;
  v << 1, nan, 3;
  rv << 1, nan, 3;

  Matrix<var,Dynamic,Dynamic> res1 = diag_pre_multiply(v,m);
  Matrix<var,Dynamic,Dynamic> res2 = diag_pre_multiply(rv,m);

  EXPECT_TRUE(boost::math::isnan(res1(0,0).val()));
  EXPECT_TRUE(boost::math::isnan(res1(1,0).val()));
  EXPECT_TRUE(boost::math::isnan(res1(1,1).val()));
  EXPECT_TRUE(boost::math::isnan(res1(1,2).val()));
  EXPECT_FLOAT_EQ(2, res1(0,1).val());
  EXPECT_FLOAT_EQ(3, res1(0,2).val());
  EXPECT_FLOAT_EQ(21, res1(2,0).val());
  EXPECT_FLOAT_EQ(24, res1(2,1).val());
  EXPECT_FLOAT_EQ(27, res1(2,2).val());

  EXPECT_TRUE(boost::math::isnan(res2(0,0).val()));
  EXPECT_TRUE(boost::math::isnan(res2(1,0).val()));
  EXPECT_TRUE(boost::math::isnan(res2(1,1).val()));
  EXPECT_TRUE(boost::math::isnan(res2(1,2).val()));
  EXPECT_FLOAT_EQ(2, res2(0,1).val());
  EXPECT_FLOAT_EQ(3, res2(0,2).val());
  EXPECT_FLOAT_EQ(21, res2(2,0).val());
  EXPECT_FLOAT_EQ(24, res2(2,1).val());
  EXPECT_FLOAT_EQ(27, res2(2,2).val());
}

TEST(MathMatrix,diagPreMultiply2_dv_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  Matrix<double,Dynamic,Dynamic> m(3,3);
  Matrix<var,Dynamic,1> v(3);
  Matrix<var,1,Dynamic> rv(3);

  m << nan, 2, 3, 4, nan, 6, 7, 8, 9;
  v << 1, nan, 3;
  rv << 1, nan, 3;

  Matrix<var,Dynamic,Dynamic> res1 = diag_pre_multiply(v,m);
  Matrix<var,Dynamic,Dynamic> res2 = diag_pre_multiply(rv,m);

  EXPECT_TRUE(boost::math::isnan(res1(0,0).val()));
  EXPECT_TRUE(boost::math::isnan(res1(1,0).val()));
  EXPECT_TRUE(boost::math::isnan(res1(1,1).val()));
  EXPECT_TRUE(boost::math::isnan(res1(1,2).val()));
  EXPECT_FLOAT_EQ(2, res1(0,1).val());
  EXPECT_FLOAT_EQ(3, res1(0,2).val());
  EXPECT_FLOAT_EQ(21, res1(2,0).val());
  EXPECT_FLOAT_EQ(24, res1(2,1).val());
  EXPECT_FLOAT_EQ(27, res1(2,2).val());

  EXPECT_TRUE(boost::math::isnan(res2(0,0).val()));
  EXPECT_TRUE(boost::math::isnan(res2(1,0).val()));
  EXPECT_TRUE(boost::math::isnan(res2(1,1).val()));
  EXPECT_TRUE(boost::math::isnan(res2(1,2).val()));
  EXPECT_FLOAT_EQ(2, res2(0,1).val());
  EXPECT_FLOAT_EQ(3, res2(0,2).val());
  EXPECT_FLOAT_EQ(21, res2(2,0).val());
  EXPECT_FLOAT_EQ(24, res2(2,1).val());
  EXPECT_FLOAT_EQ(27, res2(2,2).val());
}
