#include <stan/math/prim/scal/fun/lb_constrain.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, lb_constrain) {
  EXPECT_FLOAT_EQ(exp(-1.0) + 2.0, stan::prob::lb_constrain(-1.0,2.0));
  EXPECT_FLOAT_EQ(7.9, 
                  stan::prob::lb_constrain(7.9, -std::numeric_limits<double>::infinity()));
}
TEST(MathPrim, lb_constrain_j) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(exp(-1.0) + 2.0, stan::prob::lb_constrain(-1.0,2.0,lp));
  EXPECT_FLOAT_EQ(15.0 - 1.0, lp);

  double lp2 = 8.6;
  EXPECT_FLOAT_EQ(7.9, 
                  stan::prob::lb_constrain(7.9, -std::numeric_limits<double>::infinity(),
                                           lp2));
  EXPECT_FLOAT_EQ(8.6, lp2);
}
TEST(MathPrim, lb_rt) {
  double x = -1.0;
  double xc = stan::prob::lb_constrain(x,2.0);
  double xcf = stan::prob::lb_free(xc,2.0);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::lb_constrain(xcf,2.0);
  EXPECT_FLOAT_EQ(xc,xcfc);
}
