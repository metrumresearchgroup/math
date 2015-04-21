#include <stan/math/prim/mat/fun/ordered_free.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathPrim, ordered_free) {
  Matrix<double,Dynamic,1> y(3);
  y << -12.0, 1.1, 172.1;
  Matrix<double,Dynamic,1> x = stan::prob::ordered_free(y);
  EXPECT_EQ(y.size(),x.size());
  EXPECT_FLOAT_EQ(-12.0, x[0]);
  EXPECT_FLOAT_EQ(log(1.1 + 12.0), x[1]);
  EXPECT_FLOAT_EQ(log(172.1 - 1.1), x[2]);
}
TEST(MathPrim, ordered_free_exception) {
  Matrix<double,Dynamic,1> y(3);
  y << -0.1, 0.0, 1.0;
  EXPECT_NO_THROW(stan::prob::ordered_free(y));
  y << 0.0, 0.0, 0.0;
  EXPECT_THROW(stan::prob::ordered_free(y), std::domain_error);
  y << 0.0, 1, 0.9;
  EXPECT_THROW(stan::prob::ordered_free(y), std::domain_error);
}
