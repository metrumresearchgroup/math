#include <stan/math/prim/scal/fun/lb_constrain.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>
#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>
#include <stan/math/prim/scal/fun/lub_constrain.hpp>
#include <stan/math/prim/scal/fun/lub_free.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, lub_constrain) {
  EXPECT_FLOAT_EQ(2.0 + (5.0 - 2.0) * stan::math::inv_logit(-1.0), 
                  stan::prob::lub_constrain(-1.0,2.0,5.0));

  EXPECT_FLOAT_EQ(1.7, 
                  stan::prob::lub_constrain(1.7,
                                            -std::numeric_limits<double>::infinity(),
                                            +std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(stan::prob::lb_constrain(1.8,3.0),
                  stan::prob::lub_constrain(1.8,
                                            3.0,
                                            +std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(stan::prob::ub_constrain(1.9,-12.5),
                  stan::prob::lub_constrain(1.9,
                                            -std::numeric_limits<double>::infinity(),
                                            -12.5));
}
TEST(MathPrim, lub_constrain_j) {
  double lp = -17.0;
  double L = 2.0;
  double U = 5.0;
  double x = -1.0;
  EXPECT_FLOAT_EQ(L + (U - L) * stan::math::inv_logit(x), 
                  stan::prob::lub_constrain(x,L,U,lp));
  EXPECT_FLOAT_EQ(-17.0 + log(U - L) + log(stan::math::inv_logit(x)) 
                  + log(1.0 - stan::math::inv_logit(x)),
                  lp);

  double lp1 = -12.9;
  EXPECT_FLOAT_EQ(1.7, 
                  stan::prob::lub_constrain(1.7,
                                            -std::numeric_limits<double>::infinity(),
                                            +std::numeric_limits<double>::infinity(),
                                            lp1));
  EXPECT_FLOAT_EQ(-12.9,lp1);

  double lp2 = -19.8;
  double lp2_expected = -19.8;
  EXPECT_FLOAT_EQ(stan::prob::lb_constrain(1.8,3.0,lp2_expected),
                  stan::prob::lub_constrain(1.8,
                                            3.0,
                                            +std::numeric_limits<double>::infinity(),
                                            lp2));
  EXPECT_FLOAT_EQ(lp2_expected, lp2);

  double lp3 = -422;
  double lp3_expected = -422;
  EXPECT_FLOAT_EQ(stan::prob::ub_constrain(1.9,-12.5,lp3_expected),
                  stan::prob::lub_constrain(1.9,
                                            -std::numeric_limits<double>::infinity(),
                                            -12.5,
                                            lp3));
  EXPECT_FLOAT_EQ(lp3_expected,lp3);
  
}
TEST(MathPrim, lub_constrain_exception) {
  using stan::prob::lub_constrain;
  EXPECT_THROW(lub_constrain(5.0,1.0,1.0), std::domain_error);
  EXPECT_NO_THROW(lub_constrain(5.0,1.0,1.01));
  double lp = 12;
  EXPECT_THROW(lub_constrain(5.0,1.0,1.0,lp), std::domain_error);
  EXPECT_NO_THROW(lub_constrain(5.0,1.0,1.01,lp));
}

TEST(MathPrim, lub_rt) {
  double x = -1.0;
  double xc = stan::prob::lub_constrain(x,2.0,4.0);
  double xcf = stan::prob::lub_free(xc,2.0,4.0);
  EXPECT_FLOAT_EQ(x,xcf);
  double xcfc = stan::prob::lub_constrain(xcf,2.0,4.0);
  EXPECT_FLOAT_EQ(xc,xcfc);
}
