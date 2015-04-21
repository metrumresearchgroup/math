#include <stan/math/prim/scal/fun/identity_constrain.hpp>
#include <stan/math/prim/scal/fun/identity_free.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, identity_constrain) {
  EXPECT_FLOAT_EQ(4.0, stan::prob::identity_constrain(4.0));
}

TEST(MathPrim, identity_constrain_j) {
  double lp = 1.0;
  EXPECT_FLOAT_EQ(4.0, stan::prob::identity_constrain(4.0,lp));
  EXPECT_FLOAT_EQ(1.0,lp);
}

TEST(MathPrim, identity_rt) {
  double x = 1.2;
  double xc = stan::prob::identity_constrain(x);
  double xcf = stan::prob::identity_free(xc);
  EXPECT_FLOAT_EQ(x,xcf);

  double y = -1.0;
  double yf = stan::prob::identity_free(y);
  double yfc = stan::prob::identity_constrain(yf);
  EXPECT_FLOAT_EQ(y,yfc);
}
