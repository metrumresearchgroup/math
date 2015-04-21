#include <stan/math/prim/scal/fun/lb_free.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, lb_free) {
  EXPECT_FLOAT_EQ(log(3.0 - 2.0), stan::prob::lb_free(3.0,2.0));
  EXPECT_FLOAT_EQ(1.7, stan::prob::lb_free(1.7, -std::numeric_limits<double>::infinity()));
}
TEST(MathPrim, lb_free_exception) {
  double lb = 2.0;
  EXPECT_THROW (stan::prob::lb_free(lb - 0.01, lb), std::domain_error);
}
