#include <stan/math/rev/scal.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,a_eq_b) {
  stan::math::var a = 2.0;
  stan::math::var b = 2.0;
  EXPECT_TRUE(a == b);
  EXPECT_TRUE(b == a);
  stan::math::var c = 3.0;
  EXPECT_FALSE(a == c);
  EXPECT_FALSE(c == a);
}

TEST(AgradRev,x_eq_b) {
  double x = 2.0;
  stan::math::var b = 2.0;
  EXPECT_TRUE(x == b);
  EXPECT_TRUE(b == x);
  stan::math::var c = 3.0;
  EXPECT_FALSE(x == c);
  EXPECT_FALSE(c == x);
}

TEST(AgradRev,a_eq_y) {
  stan::math::var a = 2.0;
  double y = 2.0;
  EXPECT_TRUE(a == y);
  EXPECT_TRUE(y == a);
  double z = 3.0;
  EXPECT_FALSE(a == z);
  EXPECT_FALSE(z == a);
}

TEST(AgradRev, logical_eq_nan) {
  stan::math::var nan = std::numeric_limits<double>::quiet_NaN();
  stan::math::var a = 1.0;
  stan::math::var b = 2.0;
  double nan_dbl = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(1.0 == nan);
  EXPECT_FALSE(nan == 2.0);
  EXPECT_FALSE(nan == nan);
  EXPECT_FALSE(a == nan);
  EXPECT_FALSE(nan == b);
  EXPECT_FALSE(a == nan_dbl);
  EXPECT_FALSE(nan_dbl == b);
}
