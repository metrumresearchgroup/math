#include <stan/math/matrix/diagonal.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix, diagonal) {
  stan::math::matrix_d m0;

  using stan::math::diagonal;
  EXPECT_NO_THROW(diagonal(m0));
}
TEST(MathMatrix, diagonal_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  using stan::math::diagonal;
  using boost::math::isnan;
  
  stan::math::matrix_d m1(2,2);
  stan::math::vector_d mr;
  
  m1 << 1, nan,
        3, 6.1;
  mr = diagonal(m1);
  
  EXPECT_DOUBLE_EQ(1, mr(0));
  EXPECT_DOUBLE_EQ(6.1, mr(1));
  
  m1 << 1, 5.1,
        3, nan;
  mr = diagonal(m1);
  
  EXPECT_DOUBLE_EQ(1, mr(0));        
  EXPECT_PRED1(isnan<double>, mr(1));  
  
  m1 << nan, 5.1,
        3, nan;
  mr = diagonal(m1);
  
  EXPECT_PRED1(isnan<double>, mr(0));  
  EXPECT_PRED1(isnan<double>, mr(1));  
  
  m1 << nan, nan,
        3, nan;
  mr = diagonal(m1);
  
  EXPECT_PRED1(isnan<double>, mr(0));  
  EXPECT_PRED1(isnan<double>, mr(1));  
        
}
