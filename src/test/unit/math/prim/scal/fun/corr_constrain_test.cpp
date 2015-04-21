#include <stan/math/prim/scal/fun/corr_constrain.hpp>
#include <stan/math/prim/scal/fun/corr_free.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, corr_constrain) {
  EXPECT_FLOAT_EQ(std::tanh(-1.0), 
                  stan::prob::corr_constrain(-1.0));
}
TEST(MathPrim, corr_constrain_j) {
  double lp = -17.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(std::tanh(x), 
                  stan::prob::corr_constrain(x,lp));
  EXPECT_FLOAT_EQ(-17.0 + (log(1.0 - std::tanh(x) * std::tanh(x))),
                  lp);
}
TEST(MathPrim, corr_rt) {
  double x = -1.0;
  double xc = stan::prob::corr_constrain(x);
  double xcf = stan::prob::corr_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::corr_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}
