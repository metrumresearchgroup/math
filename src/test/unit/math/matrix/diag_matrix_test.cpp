#include <stan/math/matrix/diag_matrix.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, inverse_exception) {
  stan::math::vector_d v0;

  using stan::math::diag_matrix;
  EXPECT_NO_THROW(diag_matrix(v0));
}
TEST(MathMatrix, diag_matrix_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  using stan::math::diag_matrix;
  using boost::math::isnan;
  
  stan::math::vector_d m1(2);
  stan::math::matrix_d mr;
  
  m1 << 1.1, nan;
  mr = diag_matrix(m1);
  
  EXPECT_DOUBLE_EQ(1.1, mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(1, 1));  
  EXPECT_EQ(0, mr(0, 1));
  EXPECT_EQ(0, mr(1, 0));
  
  m1 << nan, nan;
  mr = diag_matrix(m1);
  
  EXPECT_PRED1(isnan<double>, mr(0, 0));  
  EXPECT_PRED1(isnan<double>, mr(1, 1));  
  EXPECT_EQ(0, mr(0, 1));
  EXPECT_EQ(0, mr(1, 0));
}
