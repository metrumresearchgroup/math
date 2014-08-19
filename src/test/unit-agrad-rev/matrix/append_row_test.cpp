#include <stan/math/matrix/append_row.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev.hpp>
#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/agrad/rev/functions/exp.hpp>

using stan::math::sum;
using stan::math::append_row;
using stan::agrad::matrix_v;
using stan::agrad::vector_v;
using stan::agrad::row_vector_v;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

TEST(AgradRevMatrix, append_row_matrix) {
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
  
  AVAR append_row_ab = sum(append_row(a_exp, b));

  VEC g = cgradvec(append_row_ab, x);
  
  size_t idx = 0;
  for (int i = 0; i < 2; i++)
    for (int j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(std::exp(a(i, j).val()), g[idx++]);
}

TEST(AgradRevMatrix, append_row_row_vector) {
  vector_v a(3);
  vector_v a_exp(3);
  VectorXd b(3);
  
  a << 2.0, 3.0, 9.0;
       
  b << 4.0, 3.0, 0.0;

  AVEC x;
  for (int i = 0; i < 3; ++i) {
    x.push_back(a(i));
    a_exp(i) = stan::agrad::exp(a(i));
  }
  
  AVAR append_row_ab = sum(append_row(a_exp, b));

  VEC g = cgradvec(append_row_ab, x);

  size_t idx = 0;
  for (int i = 0; i < 3; i++)
    EXPECT_FLOAT_EQ(std::exp(a(i).val()), g[idx++]);
}

TEST(AgradRevMatrix, append_row_matrix_nan) {
  matrix_v a(2,2);
  MatrixXd b(2,2);
  matrix_v b2(2,2);
  double nan = std::numeric_limits<double>::quiet_NaN();

  a << nan, 3.0,
       9.0, -1.0;
  b << nan, 3.0,
       0.0, 1.0;
  b2 << nan, 3.0,
       0.0, 1.0;

  matrix_v res1 = append_row(b, a);
  EXPECT_TRUE(boost::math::isnan(res1(0,0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(0,1).val());
  EXPECT_FLOAT_EQ(0.0, res1(1,0).val());
  EXPECT_FLOAT_EQ(1.0, res1(1,1).val());
  EXPECT_TRUE(boost::math::isnan(res1(2,0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(2,1).val());
  EXPECT_FLOAT_EQ(9.0, res1(3,0).val());
  EXPECT_FLOAT_EQ(-1.0, res1(3,1).val());

  res1 = append_row(a, b);
  EXPECT_TRUE(boost::math::isnan(res1(0,0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(0,1).val());
  EXPECT_FLOAT_EQ(9.0, res1(1,0).val());
  EXPECT_FLOAT_EQ(-1.0, res1(1,1).val());
  EXPECT_TRUE(boost::math::isnan(res1(2,0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(2,1).val());
  EXPECT_FLOAT_EQ(0.0, res1(3,0).val());
  EXPECT_FLOAT_EQ(1.0, res1(3,1).val());

  res1 = append_row(b2, a);
  EXPECT_TRUE(boost::math::isnan(res1(0,0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(0,1).val());
  EXPECT_FLOAT_EQ(0.0, res1(1,0).val());
  EXPECT_FLOAT_EQ(1.0, res1(1,1).val());
  EXPECT_TRUE(boost::math::isnan(res1(2,0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(2,1).val());
  EXPECT_FLOAT_EQ(9.0, res1(3,0).val());
  EXPECT_FLOAT_EQ(-1.0, res1(3,1).val());
}
TEST(AgradRevMatrix, append_row_row_vector_nan) {
  row_vector_v a(3);
  row_vector_v b(3);
  RowVectorXd a2(3);
  RowVectorXd b2(3);
  double nan = std::numeric_limits<double>::quiet_NaN();

  a << nan, 3.0, 9.0;
  a2 << nan, 3.0, 9.0;
  b << 4.0, nan, 0.0;
  b2 << 4.0, nan, 0.0;

  matrix_v res1 = append_row(a,b);
  EXPECT_TRUE(boost::math::isnan(res1(0,0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(0,1).val());
  EXPECT_FLOAT_EQ(9.0, res1(0,2).val());
  EXPECT_FLOAT_EQ(4.0, res1(1,0).val());
  EXPECT_TRUE(boost::math::isnan(res1(1,1).val()));
  EXPECT_FLOAT_EQ(0.0, res1(1,2).val());

  res1 = append_row(a,b2);
  EXPECT_TRUE(boost::math::isnan(res1(0,0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(0,1).val());
  EXPECT_FLOAT_EQ(9.0, res1(0,2).val());
  EXPECT_FLOAT_EQ(4.0, res1(1,0).val());
  EXPECT_TRUE(boost::math::isnan(res1(1,1).val()));
  EXPECT_FLOAT_EQ(0.0, res1(1,2).val());

  res1 = append_row(a2,b);
  EXPECT_TRUE(boost::math::isnan(res1(0,0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(0,1).val());
  EXPECT_FLOAT_EQ(9.0, res1(0,2).val());
  EXPECT_FLOAT_EQ(4.0, res1(1,0).val());
  EXPECT_TRUE(boost::math::isnan(res1(1,1).val()));
  EXPECT_FLOAT_EQ(0.0, res1(1,2).val());
}
TEST(AgradRevMatrix, append_row_vector_nan) {
  vector_v a(3);
  vector_v b(3);
  VectorXd a2(3);
  VectorXd b2(3);
  double nan = std::numeric_limits<double>::quiet_NaN();

  a << nan, 3.0, 9.0;
  a2 << nan, 3.0, 9.0;
  b << 4.0, nan, 0.0;
  b2 << 4.0, nan, 0.0;

  vector_v res1 = append_row(a,b);
  EXPECT_TRUE(boost::math::isnan(res1(0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(1).val());
  EXPECT_FLOAT_EQ(9.0, res1(2).val());
  EXPECT_FLOAT_EQ(4.0, res1(3).val());
  EXPECT_TRUE(boost::math::isnan(res1(4).val()));
  EXPECT_FLOAT_EQ(0.0, res1(5).val());

  res1 = append_row(a,b2);
  EXPECT_TRUE(boost::math::isnan(res1(0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(1).val());
  EXPECT_FLOAT_EQ(9.0, res1(2).val());
  EXPECT_FLOAT_EQ(4.0, res1(3).val());
  EXPECT_TRUE(boost::math::isnan(res1(4).val()));
  EXPECT_FLOAT_EQ(0.0, res1(5).val());

  res1 = append_row(a2,b);
  EXPECT_TRUE(boost::math::isnan(res1(0).val()));
  EXPECT_FLOAT_EQ(3.0, res1(1).val());
  EXPECT_FLOAT_EQ(9.0, res1(2).val());
  EXPECT_FLOAT_EQ(4.0, res1(3).val());
  EXPECT_TRUE(boost::math::isnan(res1(4).val()));
  EXPECT_FLOAT_EQ(0.0, res1(5).val());
}
