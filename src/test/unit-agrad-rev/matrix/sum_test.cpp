#include <stan/agrad/rev/matrix/sum.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(AgradRevMatrix, sum_vector) {
  using stan::math::sum;
  using stan::math::vector_d;
  using stan::agrad::vector_v;

  vector_d d(6);
  vector_v v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  AVAR output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val());
}
TEST(AgradRevMatrix, sum_rowvector) {
  using stan::math::sum;
  using stan::math::row_vector_d;
  using stan::agrad::row_vector_v;

  row_vector_d d(6);
  row_vector_v v(6);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  AVAR output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val());
}
TEST(AgradRevMatrix, sum_matrix) {
  using stan::math::sum;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

  matrix_d d(2, 3);
  matrix_v v(2, 3);
  
  d << 1, 2, 3, 4, 5, 6;
  v << 1, 2, 3, 4, 5, 6;
  
  AVAR output;
  output = sum(d);
  EXPECT_FLOAT_EQ(21.0, output.val());
                   
  output = sum(v);
  EXPECT_FLOAT_EQ(21.0, output.val());

  d.resize(0, 0);
  v.resize(0, 0);
  EXPECT_FLOAT_EQ(0.0, sum(d));
  EXPECT_FLOAT_EQ(0.0, sum(v).val());
}

TEST(AgradRevMatrix, sum_vector_nan) {
  using stan::math::sum;
  using stan::agrad::vector_v;
  double nan = std::numeric_limits<double>::quiet_NaN();

  vector_v v(6);
  v << 1, 2, nan, 4, nan, 6;
  
  AVAR output;
  output = sum(v);
  EXPECT_TRUE(boost::math::isnan(output.val()));
}
TEST(AgradRevMatrix, sum_rowvector_nan) {
  using stan::math::sum;
  using stan::agrad::row_vector_v;
  double nan = std::numeric_limits<double>::quiet_NaN();

  row_vector_v v(6);
  v << 1, 2, nan, 4, nan, 6;
  
  AVAR output;
  output = sum(v);
  EXPECT_TRUE(boost::math::isnan(output.val()));
}
TEST(AgradRevMatrix, sum_matrix_nan) {
  using stan::math::sum;
  using stan::agrad::matrix_v;
  double nan = std::numeric_limits<double>::quiet_NaN();

  matrix_v v(2, 3);
  v << 1, 2, nan, 4, nan, 6;
  
  AVAR output;
  output = sum(v);
  EXPECT_TRUE(boost::math::isnan(output.val()));
}
