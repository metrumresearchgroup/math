#include <stan/math/matrix/exp.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/functions/exp.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(AgradRevMatrix, exp_matrix) {
  using stan::math::exp;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

  matrix_d expected_output(2,2);
  matrix_v mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::exp(1), std::exp(2), std::exp(3), std::exp(4);
  output = exp(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val());
}

TEST(AgradRevMatrix, exp_matrix_nan) {
  using stan::math::exp;
  using stan::agrad::matrix_v;
  double nan = std::numeric_limits<double>::quiet_NaN();

  matrix_v mv(2,2), output;

  mv << 1, nan, 3, 4;
  output = exp(mv);

  EXPECT_FLOAT_EQ(std::exp(1.0), output(0,0).val());
  EXPECT_TRUE(boost::math::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ(std::exp(3.0), output(1,0).val());
  EXPECT_FLOAT_EQ(std::exp(4.0), output(1,1).val());
}
