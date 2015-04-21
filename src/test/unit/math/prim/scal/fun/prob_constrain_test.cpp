#include <stan/math/prim/scal/fun/prob_constrain.hpp>
#include <stan/math/prim/scal/fun/prob_free.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, prob_constrain) {
  EXPECT_FLOAT_EQ(stan::math::inv_logit(-1.0), 
                  stan::prob::prob_constrain(-1.0));
}
TEST(MathPrim, prob_constrain_j) {
  double lp = -17.0;
  double L = 0.0;
  double U = 1.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(L + (U - L) * stan::math::inv_logit(x), 
                  stan::prob::prob_constrain(x,lp));
  EXPECT_FLOAT_EQ(-17.0 + log(U - L) + log(stan::math::inv_logit(x)) 
                  + log(1.0 - stan::math::inv_logit(x)),
                  lp);
}

TEST(MathPrim, prob_rt) {
  double x = -1.0;
  double xc = stan::prob::prob_constrain(x);
  double xcf = stan::prob::prob_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::prob_constrain(xcf);
  EXPECT_FLOAT_EQ(xc,xcfc);
}
