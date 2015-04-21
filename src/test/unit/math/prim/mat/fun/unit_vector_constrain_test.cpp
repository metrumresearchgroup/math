#include <stan/math/prim/mat/fun/unit_vector_constrain.hpp>
#include <stan/math/prim/mat/fun/unit_vector_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_free.hpp>
#include <test/unit/util.hpp>

#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;


TEST(MathPrim, unit_vector_rt0) {
  Matrix<double,Dynamic,1> x(4);
  x << 0.0, 0.0, 0.0, 0.0;
  Matrix<double,Dynamic,1> y = stan::prob::unit_vector_constrain(x);
  EXPECT_NEAR(0, y(0), 1e-8);
  EXPECT_NEAR(0, y(1), 1e-8);
  EXPECT_NEAR(0, y(2), 1e-8);
  EXPECT_NEAR(0, y(3), 1e-8);
  EXPECT_NEAR(1.0, y(4), 1e-8);

  Matrix<double,Dynamic,1> xrt = stan::prob::unit_vector_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_NEAR(x[i],xrt[i],1E-10);
  }
}
TEST(MathPrim, unit_vector_rt) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 1.0;
  Matrix<double,Dynamic,1> y = stan::prob::unit_vector_constrain(x);
  Matrix<double,Dynamic,1> xrt = stan::prob::unit_vector_free(y);
  EXPECT_EQ(x.size()+1,y.size());
  EXPECT_EQ(x.size(),xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i],xrt[i]) << "error in component " << i;
  }
}
TEST(MathPrim, unit_vector_match) {
  Matrix<double,Dynamic,1> x(3);
  x << 1.0, -1.0, 2.0;
  double lp;
  Matrix<double,Dynamic,1> y = stan::prob::unit_vector_constrain(x);
  Matrix<double,Dynamic,1> y2 = stan::prob::unit_vector_constrain(x,lp);

  EXPECT_EQ(4,y.size());
  EXPECT_EQ(4,y2.size());
  for (int i = 0; i < x.size(); ++i)
    EXPECT_FLOAT_EQ(y[i],y2[i]) << "error in component " << i;
}
