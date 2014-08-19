#include <stan/math/matrix/row.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix,row) {
  stan::math::matrix_d m(3,4);
  m << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12;
  stan::math::row_vector_d c = m.row(1);
  stan::math::row_vector_d c2 = stan::math::row(m,2);
  EXPECT_EQ(4,c.size());
  EXPECT_EQ(4,c2.size());
  for (size_t i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(c[i],c2[i]);
}

TEST(MathMatrix, row_exception) {
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  
  using stan::math::row;

  EXPECT_THROW(row(m1,5),std::domain_error);
  EXPECT_THROW(row(m1,0),std::domain_error);
}

TEST(MathMatrix,row_nan) {
  using stan::math::row;
  using stan::math::matrix_d;
  using stan::math::vector_d;
  double nan = std::numeric_limits<double>::quiet_NaN();

  matrix_d y(2,3);
  y << 1, 2, nan, 4, 5, 6;
  vector_d z = row(y,1);
  EXPECT_EQ(3,z.size());
  EXPECT_FLOAT_EQ(1.0,z[0]);
  EXPECT_FLOAT_EQ(2.0,z[1]);
  EXPECT_TRUE(boost::math::isnan(z[2]));
}
