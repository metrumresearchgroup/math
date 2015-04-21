#include <stan/math/prim/mat/fun/positive_ordered_constrain.hpp>
#include <stan/math/prim/mat/fun/positive_ordered_free.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathPrim, positive_ordered_constrain) {
  Matrix<double,Dynamic,1> x(3);
  x << -15.0, -2.0, -5.0;
  Matrix<double,Dynamic,1> y = stan::prob::positive_ordered_constrain(x);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(exp(-15.0), y[0]);
  EXPECT_EQ(exp(-15.0) + exp(-2.0), y[1]);
  EXPECT_EQ(exp(-15.0) + exp(-2.0) + exp(-5.0), y[2]);
}
TEST(MathPrim, positive_ordered_constrain_j) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -2.0, -5.0;
  double lp = -152.1;
  Matrix<double,Dynamic,1> y = stan::prob::positive_ordered_constrain(x,lp);
  EXPECT_EQ(x.size(), y.size());
  EXPECT_EQ(exp(1.0), y[0]);
  EXPECT_EQ(exp(1.0) + exp(-2.0), y[1]);
  EXPECT_EQ(exp(1.0) + exp(-2.0) + exp(-5.0), y[2]);
  EXPECT_EQ(-152.1 + 1.0 - 2.0 - 5.0,lp);
}
TEST(MathPrim, positive_ordered_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << -1.0, 8.0, -3.9;
  Matrix<double,Dynamic,1> y = stan::prob::positive_ordered_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::prob::positive_ordered_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
