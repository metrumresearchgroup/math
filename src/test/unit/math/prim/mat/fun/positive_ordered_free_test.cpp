#include <stan/math/prim/mat/fun/positive_ordered_free.hpp>
#include <test/unit/util.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathPrim, positive_ordered_free) {
  Matrix<double,Dynamic,1> y(3);
  y << 0.12, 1.1, 172.1;
  Matrix<double,Dynamic,1> x = stan::prob::positive_ordered_free(y);
  EXPECT_EQ(y.size(),x.size());
  EXPECT_FLOAT_EQ(log(0.12), x[0]);
  EXPECT_FLOAT_EQ(log(1.1 - 0.12), x[1]);
  EXPECT_FLOAT_EQ(log(172.1 - 1.1), x[2]);
}
TEST(MathPrim, positive_ordered_free_exception) {
  Matrix<double,Dynamic,1> y(3);
  y << -0.1, 0.0, 1.0;
  EXPECT_THROW(stan::prob::positive_ordered_free(y), std::domain_error);
  y << 0.0, 0.0, 0.0;
  EXPECT_THROW(stan::prob::positive_ordered_free(y), std::domain_error);
  y << 0.0, 1, 0.9;
  EXPECT_THROW(stan::prob::positive_ordered_free(y), std::domain_error);
}
