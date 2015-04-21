#include <stan/math/prim/mat/fun/simplex_constrain.hpp>
#include <stan/math/prim/mat/fun/simplex_free.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathPrim, simplex_rt0) {
  Matrix<double,Dynamic,1> x(4);
  x << 0.0, 0.0, 0.0, 0.0;
  Matrix<double,Dynamic,1> y = stan::prob::simplex_constrain(x);
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(0));
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(1));
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(2));
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(3));
  EXPECT_FLOAT_EQ(1.0 / 5.0, y(4));

  Matrix<double,Dynamic,1> xrt = stan::prob::simplex_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(x[i],xrt[i],1E-10);
  }
}
TEST(MathPrim, simplex_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 2.0;
  Matrix<double,Dynamic,1> y = stan::prob::simplex_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::prob::simplex_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i],xrt[i]);
  }
}
TEST(MathPrim, simplex_match) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 2.0;
  double lp;
  Matrix<double,Dynamic,1> y = stan::prob::simplex_constrain(x);
  Matrix<double,Dynamic,1> y2 = stan::prob::simplex_constrain(x,lp);

  EXPECT_EQ(4,y.size());
  EXPECT_EQ(4,y2.size());
  for (int i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(y[i],y2[i]);
}
