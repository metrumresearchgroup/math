#include <stan/math/rev/scal.hpp>
#include <gtest/gtest.h>

TEST(AgradRev,not_a) {
  stan::math::var a(6.0);
  EXPECT_EQ(0, !a);
  stan::math::var b(0.0);
  EXPECT_EQ(1, !b);
}

TEST(AgradRev,not_nan) {
  stan::math::var nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(!nan);
}
