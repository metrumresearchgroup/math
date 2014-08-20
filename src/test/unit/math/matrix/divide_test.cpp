#include <stan/math/matrix/divide.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix, divide) {
  stan::math::vector_d v0;
  stan::math::row_vector_d rv0;
  stan::math::matrix_d m0;

  using stan::math::divide;
  EXPECT_NO_THROW(divide(v0,2.0));
  EXPECT_NO_THROW(divide(rv0,2.0));
  EXPECT_NO_THROW(divide(m0,2.0));
}

TEST(MathMatrix, divide_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  stan::math::matrix_d m1(2,2);
  m1 << 1, nan,
        3, 6.15;
        
  stan::math::matrix_d m2(2,2);
  m2 << 1, 100,
        nan, 4.9;
        
  stan::math::matrix_d m3(2,2);
  m3 << 10.1, 100,
        1, 0;
        
  stan::math::matrix_d mr;

  using stan::math::divide;
  using boost::math::isnan;
  
  mr = divide(m1, 0.1);
  
  EXPECT_DOUBLE_EQ(10, mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_DOUBLE_EQ(30, mr(1, 0));
  EXPECT_DOUBLE_EQ(61.5, mr(1, 1));
  
  expect_matrix_is_nan(divide(m1, nan));
  expect_matrix_is_nan(divide(m2, nan));
  expect_matrix_is_nan(divide(m3, nan));
}
