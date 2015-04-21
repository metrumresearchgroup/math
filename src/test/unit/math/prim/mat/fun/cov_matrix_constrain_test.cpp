#include <stan/math/prim/mat/fun/cov_matrix_constrain.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_free.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathPrim, cov_matrix_rt) {
  unsigned int K = 4;
  unsigned int K_choose_2 = 6; 
  Matrix<double,Dynamic,1> x(K_choose_2 + K);
  x << -1.0, 2.0, 0.0, 1.0, 3.0, -1.5,
    1.0, 2.0, -1.5, 2.5;
  Matrix<double,Dynamic,Dynamic> y = stan::prob::cov_matrix_constrain(x,K);
  Matrix<double,Dynamic,1> xrt = stan::prob::cov_matrix_free(y);
  EXPECT_EQ(x.size(), xrt.size());
  for (int i = 0; i < x.size(); ++i) {
    EXPECT_FLOAT_EQ(x[i], xrt[i]);
  }
}
TEST(MathPrim, cov_matrix_constrain_exception) {
  Matrix<double,Dynamic,1> x(7);
  int K = 12;
  EXPECT_THROW(stan::prob::cov_matrix_constrain(x,K), std::domain_error);
}
