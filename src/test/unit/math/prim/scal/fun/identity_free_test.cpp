#include <stan/math/prim/scal/fun/identity_free.hpp>
#include <gtest/gtest.h>

TEST(prob_transform,identity_free) {
  EXPECT_FLOAT_EQ(4.0, stan::prob::identity_free(4.0));
}
