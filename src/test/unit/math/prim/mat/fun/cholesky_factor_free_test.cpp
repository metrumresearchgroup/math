#include <stan/math/prim/mat/fun/cholesky_factor_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_free.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathPrim, choleskyFactorFreeError) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::prob::cholesky_factor_free;

  Matrix<double,Dynamic,Dynamic> y(1,1);
  y.resize(1,1);
  y << -2;
  EXPECT_THROW(cholesky_factor_free(y),std::domain_error);

  y.resize(2,2);
  y << 1, 2, 3, 4;
  EXPECT_THROW(cholesky_factor_free(y),std::domain_error);

  y.resize(2,3);
  y << 1, 0, 0,
    2, 3, 0;
  EXPECT_THROW(cholesky_factor_free(y),std::domain_error);
}
