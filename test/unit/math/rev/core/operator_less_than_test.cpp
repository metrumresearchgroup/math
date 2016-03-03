#include <stan/math/rev/scal.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,a_lt_b) {
  stan::math::var a = 5.0;
  stan::math::var b = 6.0;
  EXPECT_TRUE(a < b);
  EXPECT_FALSE(b < a);
  stan::math::var c = 6.0;
  EXPECT_FALSE(b < c);
  EXPECT_FALSE(c < b);
}

TEST(AgradRev,a_lt_y) {
  stan::math::var a = 5.0;
  double y = 6.0;
  EXPECT_TRUE(a < y);
  EXPECT_FALSE(y < a);
  stan::math::var b = 6.0;
  EXPECT_FALSE(b < y);
  EXPECT_FALSE(y < b);
}

TEST(AgradRev,x_lt_b) {
  double x = 5.0;
  stan::math::var b = 6.0;
  EXPECT_TRUE(x < b);
  EXPECT_FALSE(b < x);
  double y = 6.0;
  EXPECT_FALSE(b < y);
  EXPECT_FALSE(y < b);
}

TEST(AgradRev, logical_lt_nan) {
  stan::math::var nan = std::numeric_limits<double>::quiet_NaN();
  stan::math::var a = 1.0;
  stan::math::var b = 2.0;
  double nan_dbl = std::numeric_limits<double>::quiet_NaN();

  EXPECT_FALSE(1.0 < nan);
  EXPECT_FALSE(nan < 2.0);
  EXPECT_FALSE(nan < nan);
  EXPECT_FALSE(a < nan);
  EXPECT_FALSE(nan < b);
  EXPECT_FALSE(a < nan_dbl);
  EXPECT_FALSE(nan_dbl < b);
}
