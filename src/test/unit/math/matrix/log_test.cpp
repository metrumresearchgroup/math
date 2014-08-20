#include <stan/math/matrix/log.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

// log tests
TEST(MathMatrix, log) {
  using stan::math::log;
  stan::math::matrix_d logected_output(2,2);
  stan::math::matrix_d mv(2,2), output;
  int i,j;

  mv << 1, 2, 3, 4;
  logected_output << std::log(1), std::log(2), std::log(3), std::log(4);
  output = log(mv);

  for (i = 0; i < 2; i++)
    for (j = 0; j < 2; j++)
      EXPECT_FLOAT_EQ(logected_output(i,j), output(i,j));
}

TEST(MathMatrix, log_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  stan::math::matrix_d m1(2,2);
  m1 << 1, nan,
        3, 6;
        
  stan::math::matrix_d mr;

  using stan::math::log;
  using boost::math::isnan;
  
  mr = log(m1);
  
  EXPECT_DOUBLE_EQ(std::log(1), mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_DOUBLE_EQ(std::log(3), mr(1, 0));
  EXPECT_DOUBLE_EQ(std::log(6), mr(1, 1));
}
