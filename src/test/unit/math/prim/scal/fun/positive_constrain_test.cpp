#include <stan/math/prim/scal/fun/positive_constrain.hpp>
#include <stan/math/prim/scal/fun/positive_free.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, positive_constrain) {
  EXPECT_FLOAT_EQ(exp(-1.0), stan::prob::positive_constrain(-1.0));
}
TEST(MathPrim, positive_constrain_j) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(exp(-1.0), stan::prob::positive_constrain(-1.0,lp));
  EXPECT_FLOAT_EQ(15.0 - 1.0, lp);
}
TEST(MathPrim, positive_rt) {
  double x = -1.0;
  double xc = stan::prob::positive_constrain(x);
  double xcf = stan::prob::positive_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::positive_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}

