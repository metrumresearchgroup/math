#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, ub_constrain) {
  EXPECT_FLOAT_EQ(2.0 - exp(-1.0), stan::prob::ub_constrain(-1.0,2.0));
  EXPECT_FLOAT_EQ(1.7, 
                  stan::prob::ub_constrain(1.7, 
                                           std::numeric_limits<double>::infinity()));
}
TEST(prob_transform, ub_constrain_jacobian) {
  double lp = 15.0;
  EXPECT_FLOAT_EQ(2.0 - exp(-1.0), stan::prob::ub_constrain(-1.0,2.0,lp));
  EXPECT_FLOAT_EQ(15.0 - 1.0, lp);

  double lp2 = 1.87;
  EXPECT_FLOAT_EQ(-5.2, stan::prob::ub_constrain(-5.2,
                                                 std::numeric_limits<double>::infinity(),
                                                 lp2));
  EXPECT_FLOAT_EQ(1.87,lp2);
}

TEST(prob_transform, ub_rt) {
  double x = -1.0;
  double xc = stan::prob::ub_constrain(x,2.0);
  double xcf = stan::prob::ub_free(xc,2.0);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::ub_constrain(xcf,2.0);
  EXPECT_FLOAT_EQ(xc,xcfc);
}

