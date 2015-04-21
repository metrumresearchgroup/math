#include <stan/math/prim/mat/fun/factor_U.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathPrim, factorU) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using Eigen::Array;
  using stan::prob::factor_U;
  int K = 3;
  Matrix<double,Dynamic,Dynamic> U(K,K);
  U << 
    1.0, -0.25, 0.75,
    0.0,  1.0,  0.487950036474267,
    0.0,  0.0,  1.0;
  Eigen::Array<double,Dynamic,1> CPCs( (K * (K - 1)) / 2);
  CPCs << 10, 100, 1000;
  factor_U(U, CPCs);
  // test that function doesn't resize itself
  EXPECT_EQ( (K * (K - 1)) / 2, CPCs.size());
  for (int i = 0; i < CPCs.size(); ++i)
    EXPECT_LE(std::tanh(std::fabs(CPCs(i))), 1.0) << CPCs(i);
}
