#include <stan/math/matrix/append_col.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/agrad/rev/functions/exp.hpp>

using stan::math::sum;
using stan::math::append_col;
using stan::agrad::matrix_v;
using stan::agrad::row_vector_v;
using Eigen::MatrixXd;
using Eigen::RowVectorXd;

TEST(AgradRevMatrix, append_col_matrix) {
  matrix_v a(2,2);
  matrix_v a_exp(2,2);
  MatrixXd b(2,2);
  
  a << 2.0, 3.0,
       9.0, -1.0;
       
  b << 4.0, 3.0,
       0.0, 1.0;

  AVEC x;
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      x.push_back(a(i,j));
      a_exp(i, j) = stan::agrad::exp(a(i, j));
    }
  }
  
  AVAR append_col_ab = sum(append_col(a_exp, b));

  VEC g = cgradvec(append_col_ab, x);
  
  size_t idx = 0;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(std::exp(a(i, j).val()), g[idx++]);
}

TEST(AgradRevMatrix, append_col_row_vector) {
  row_vector_v a(3);
  row_vector_v a_exp(3);
  RowVectorXd b(2);
  
  a << 2.0, 3.0, 9.0;
       
  b << 4.0, 3.0, 0.0;

  AVEC x;
  for (int i = 0; i < 3; ++i) {
    x.push_back(a(i));
    a_exp(i) = stan::agrad::exp(a(i));
  }
  
  AVAR append_col_ab = sum(append_col(a_exp, b));

  VEC g = cgradvec(append_col_ab, x);
  
  size_t idx = 0;
  for (int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(std::exp(a(i).val()), g[idx++]);
}

TEST(AgradRevMatrix, append_col_matrix_nan) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using boost::math::isnan;
  double nan = std::numeric_limits<double>::quiet_NaN();

  matrix_d m1(3,2);
  matrix_d m2(3,2);
  matrix_v m1_v(3,2);
  matrix_v m2_v(3,2);

  m1 << 1, nan, 3, 4.1, nan, 6;
  m2 << 10.1, 100, nan, 0, -10, -12;
  m1_v << 1, nan, 3, 4.1, nan, 6;
  m2_v << 10.1, 100, nan, 0, -10, -12;
        
  matrix_v res1 = append_col(m1,m2_v);
  matrix_v res2 = append_col(m1_v,m2);
  matrix_v res3 = append_col(m1_v,m2_v);

  EXPECT_FLOAT_EQ(1, res1(0, 0).val());
  EXPECT_PRED1(isnan<double>, res1(0, 1).val());
  EXPECT_FLOAT_EQ(3, res1(1, 0).val());
  EXPECT_FLOAT_EQ(4.1, res1(1, 1).val());
  EXPECT_PRED1(isnan<double>, res1(2, 0).val());  
  EXPECT_FLOAT_EQ(6, res1(2, 1).val());
  EXPECT_FLOAT_EQ(10.1, res1(0, 2).val());
  EXPECT_FLOAT_EQ(100, res1(0, 3).val());
  EXPECT_PRED1(isnan<double>, res1(1, 2).val());
  EXPECT_FLOAT_EQ(0, res1(1, 3).val());
  EXPECT_FLOAT_EQ(-10, res1(2, 2).val());
  EXPECT_FLOAT_EQ(-12, res1(2, 3).val());

  EXPECT_FLOAT_EQ(1, res2(0, 0).val());
  EXPECT_PRED1(isnan<double>, res2(0, 1).val());
  EXPECT_FLOAT_EQ(3, res2(1, 0).val());
  EXPECT_FLOAT_EQ(4.1, res2(1, 1).val());
  EXPECT_PRED1(isnan<double>, res2(2, 0).val());  
  EXPECT_FLOAT_EQ(6, res2(2, 1).val());
  EXPECT_FLOAT_EQ(10.1, res2(0, 2).val());
  EXPECT_FLOAT_EQ(100, res2(0, 3).val());
  EXPECT_PRED1(isnan<double>, res2(1, 2).val());
  EXPECT_FLOAT_EQ(0, res2(1, 3).val());
  EXPECT_FLOAT_EQ(-10, res2(2, 2).val());
  EXPECT_FLOAT_EQ(-12, res2(2, 3).val());

  EXPECT_FLOAT_EQ(1, res3(0, 0).val());
  EXPECT_PRED1(isnan<double>, res3(0, 1).val());
  EXPECT_FLOAT_EQ(3, res3(1, 0).val());
  EXPECT_FLOAT_EQ(4.1, res3(1, 1).val());
  EXPECT_PRED1(isnan<double>, res3(2, 0).val());  
  EXPECT_FLOAT_EQ(6, res3(2, 1).val());
  EXPECT_FLOAT_EQ(10.1, res3(0, 2).val());
  EXPECT_FLOAT_EQ(100, res3(0, 3).val());
  EXPECT_PRED1(isnan<double>, res3(1, 2).val());
  EXPECT_FLOAT_EQ(0, res3(1, 3).val());
  EXPECT_FLOAT_EQ(-10.0, res3(2, 2).val());
  EXPECT_FLOAT_EQ(-12, res3(2, 3).val());
}

TEST(AgradRevMatrix, append_col_row_vector_nan) {
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;
  using boost::math::isnan;
  double nan = std::numeric_limits<double>::quiet_NaN();

  row_vector_d m1(3);
  row_vector_d m2(3);
  row_vector_v m1_v(3);
  row_vector_v m2_v(3);

  m1 << 1, nan, 3;
  m2 << 10.1, 100, nan;
  m1_v << 1, nan, 3;
  m2_v << 10.1, 100, nan;
        
  row_vector_v res1 = append_col(m1,m2_v);
  row_vector_v res2 = append_col(m1_v,m2);
  row_vector_v res3 = append_col(m1_v,m2_v);

  EXPECT_FLOAT_EQ(1, res1(0).val());
  EXPECT_PRED1(isnan<double>, res1(1).val());
  EXPECT_FLOAT_EQ(3, res1(2).val());
  EXPECT_FLOAT_EQ(10.1, res1(3).val());
  EXPECT_FLOAT_EQ(100, res1(4).val());
  EXPECT_PRED1(isnan<double>, res1(5).val());

  EXPECT_FLOAT_EQ(1, res2(0).val());
  EXPECT_PRED1(isnan<double>, res2(1).val());
  EXPECT_FLOAT_EQ(3, res2(2).val());
  EXPECT_FLOAT_EQ(10.1, res2(3).val());
  EXPECT_FLOAT_EQ(100, res2(4).val());
  EXPECT_PRED1(isnan<double>, res2(5).val());

  EXPECT_FLOAT_EQ(1, res3(0).val());
  EXPECT_PRED1(isnan<double>, res3(1).val());
  EXPECT_FLOAT_EQ(3, res3(2).val());
  EXPECT_FLOAT_EQ(10.1, res3(3).val());
  EXPECT_FLOAT_EQ(100, res3(4).val());
  EXPECT_PRED1(isnan<double>, res3(5).val());
}

TEST(AgradRevMatrix, append_col_vector_nan) {
  using stan::math::vector_d;
  using stan::agrad::matrix_v;
  using stan::agrad::vector_v;
  using boost::math::isnan;
  double nan = std::numeric_limits<double>::quiet_NaN();

  vector_d m1(3);
  vector_d m2(3);
  vector_v m1_v(3);
  vector_v m2_v(3);

  m1 << 1, nan, 3;
  m2 << 10.1, 100, nan;
  m1_v << 1, nan, 3;
  m2_v << 10.1, 100, nan;
        
  matrix_v res1 = append_col(m1,m2_v);
  matrix_v res2 = append_col(m1_v,m2);
  matrix_v res3 = append_col(m1_v,m2_v);

  EXPECT_FLOAT_EQ(1, res1(0,0).val());
  EXPECT_PRED1(isnan<double>, res1(1,0).val());
  EXPECT_FLOAT_EQ(3, res1(2,0).val());
  EXPECT_FLOAT_EQ(10.1, res1(0,1).val());
  EXPECT_FLOAT_EQ(100.0, res1(1,1).val());
  EXPECT_PRED1(isnan<double>, res1(2,1).val());

  EXPECT_FLOAT_EQ(1, res2(0,0).val());
  EXPECT_PRED1(isnan<double>, res2(1,0).val());
  EXPECT_FLOAT_EQ(3, res2(2,0).val());
  EXPECT_FLOAT_EQ(10.1, res2(0,1).val());
  EXPECT_FLOAT_EQ(100.0, res2(1,1).val());
  EXPECT_PRED1(isnan<double>, res2(2,1).val());

  EXPECT_FLOAT_EQ(1, res3(0,0).val());
  EXPECT_PRED1(isnan<double>, res3(1,0).val());
  EXPECT_FLOAT_EQ(3, res3(2,0).val());
  EXPECT_FLOAT_EQ(10.1, res3(0,1).val());
  EXPECT_FLOAT_EQ(100.0, res3(1,1).val());
  EXPECT_PRED1(isnan<double>, res3(2,1).val());
}
