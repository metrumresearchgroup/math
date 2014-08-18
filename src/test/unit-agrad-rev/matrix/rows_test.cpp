#include <stan/math/matrix/rows.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>

TEST(AgradRevMatrix,rows_vector) {
  using stan::agrad::vector_v;
  using stan::agrad::row_vector_v;
  using stan::math::rows;

  vector_v v(5);
  v << 0, 1, 2, 3, 4;
  EXPECT_EQ(5U, rows(v));
  
  v.resize(0);
  EXPECT_EQ(0U, rows(v));
}
TEST(AgradRevMatrix,rows_rowvector) {
  using stan::agrad::row_vector_v;
  using stan::math::rows;

  row_vector_v rv(5);
  rv << 0, 1, 2, 3, 4;
  EXPECT_EQ(1U, rows(rv));

  rv.resize(0);
  EXPECT_EQ(1U, rows(rv));
}
TEST(AgradRevMatrix,rows_matrix) {
  using stan::agrad::matrix_v;
  using stan::math::rows;

  matrix_v m(2,3);
  m << 0, 1, 2, 3, 4, 5;
  EXPECT_EQ(2U, rows(m));
  
  m.resize(0,2);
  EXPECT_EQ(0U, rows(m));
}

TEST(AgradRevMatrix,rows_vector_nan) {
  using stan::agrad::vector_v;
  using stan::agrad::row_vector_v;
  using stan::math::rows;
  double nan = std::numeric_limits<double>::quiet_NaN();

  vector_v v(5);
  v << 0, nan, 2, nan, 4;
  EXPECT_EQ(5U, rows(v));
  
  v.resize(0);
  EXPECT_EQ(0U, rows(v));
}
TEST(AgradRevMatrix,rows_rowvector_nan) {
  using stan::agrad::row_vector_v;
  using stan::math::rows;
  double nan = std::numeric_limits<double>::quiet_NaN();

  row_vector_v rv(5);
  rv << 0, nan, 2, nan, 4;
  EXPECT_EQ(1U, rows(rv));

  rv.resize(0);
  EXPECT_EQ(1U, rows(rv));
}
TEST(AgradRevMatrix,rows_matrix_nan) {
  using stan::agrad::matrix_v;
  using stan::math::rows;

  matrix_v m(2,3);
  m << 0, nan, 2, 3, nan, 5;
  EXPECT_EQ(2U, rows(m));
  
  m.resize(0,2);
  EXPECT_EQ(0U, rows(m));
}
