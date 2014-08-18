#include <stan/math/matrix/log.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>

TEST(AgradRevMatrix, log_matrix) {
  using stan::math::log;
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;

  matrix_d expected_output(2,2);
  matrix_v mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  expected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(expected_output(i,j), output(i,j).val());
}

TEST(AgradRevMatrix, log_matrix_nan) {
  using stan::math::log;
  using stan::agrad::matrix_v;
  double nan = std::numeric_limits<double>::quiet_NaN();

  matrix_v mv(2,2), output;

  mv << 1, nan, 3, 4;
  output = log(mv);

  EXPECT_FLOAT_EQ(std::log(1.0), output(0,0).val());
  EXPECT_TRUE(boost::math::isnan(output(0,1).val()));
  EXPECT_FLOAT_EQ(std::log(3.0), output(1,0).val());
  EXPECT_FLOAT_EQ(std::log(4.0), output(1,1).val());
}
