#include <stan/math/matrix/minus.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, minus) {
  stan::math::vector_d v0;
  stan::math::row_vector_d rv0;
  stan::math::matrix_d m0;

  EXPECT_EQ(0,stan::math::minus(v0).size());
  EXPECT_EQ(0,stan::math::minus(rv0).size());
  EXPECT_EQ(0,stan::math::minus(m0).size());
}

TEST(MathMatrix, minus_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  stan::math::matrix_d m1(2,2);
  m1 << 1, nan,
        3, 6;
        
  stan::math::matrix_d mr;

  using stan::math::minus;
  using boost::math::isnan;
  
  mr = minus(m1);
  
  EXPECT_DOUBLE_EQ(-1, mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_DOUBLE_EQ(-3, mr(1, 0));
  EXPECT_DOUBLE_EQ(-6, mr(1, 1));
}
