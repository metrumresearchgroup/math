#include <stan/math/prim/scal/fun/positive_free.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

TEST(MathPrim, positive_free) {
  EXPECT_FLOAT_EQ(log(0.5), stan::prob::positive_free(0.5));
}
TEST(MathPrim, positive_free_exception) {
  EXPECT_THROW (stan::prob::positive_free(-1.0), std::domain_error);
}
