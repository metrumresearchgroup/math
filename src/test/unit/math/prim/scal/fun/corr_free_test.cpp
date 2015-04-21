#include <stan/math/prim/scal/fun/corr_free.hpp>
#include <gtest/gtest.h>

TEST(prob_transform, corr_free) {
  EXPECT_FLOAT_EQ(atanh(-0.4), 0.5 * std::log((1.0 + -0.4)/(1.0 - -0.4)));
  double y = -0.4;
  EXPECT_FLOAT_EQ(atanh(y),
                  stan::prob::corr_free(y));
}
