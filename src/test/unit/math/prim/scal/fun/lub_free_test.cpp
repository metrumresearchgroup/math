#include <stan/math/prim/scal/fun/lub_free.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, lub_free) {
  double L = -10.0;
  double U = 27.0;
  double y = 3.0;
  EXPECT_FLOAT_EQ(stan::math::logit((y - L) / (U - L)),
                  stan::prob::lub_free(y,L,U));
  
  EXPECT_FLOAT_EQ(14.2,
                  stan::prob::lub_free(14.2,
                                       -std::numeric_limits<double>::infinity(),
                                       std::numeric_limits<double>::infinity()));
  EXPECT_FLOAT_EQ(stan::prob::ub_free(-18.3,7.6),
                  stan::prob::lub_free(-18.3,
                                       -std::numeric_limits<double>::infinity(),
                                       7.6));
  EXPECT_FLOAT_EQ(stan::prob::lb_free(763.9, -3122.2),
                  stan::prob::lub_free(763.9,
                                       -3122.2,
                                       std::numeric_limits<double>::infinity()));
}
TEST(MathPrim, lub_free_exception) {
  double L = -10.0;
  double U = 27.0;
  EXPECT_THROW(stan::prob::lub_free (L-0.01,L,U), std::domain_error);
  EXPECT_THROW(stan::prob::lub_free (U+0.01,L,U), std::domain_error);

  EXPECT_THROW(stan::prob::lub_free ((L+U)/2,U,L), std::domain_error);
}
