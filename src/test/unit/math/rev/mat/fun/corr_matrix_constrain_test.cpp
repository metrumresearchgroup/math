#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/sqrt.hpp>
#include <stan/math/rev/scal/fun/tanh.hpp>
#include <stan/math/rev/mat/fun/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/rev/mat/fun/sum.hpp>
#include <stan/math/prim/mat/fun/determinant.hpp>
#include <stan/math/prim/mat/fun/corr_matrix_constrain.hpp>
#include <test/unit/math/rev/mat/fun/jacobian.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathRev, corr_matrix_constrain_jacobian) {
  using stan::agrad::var;
  using stan::math::determinant;
  using std::log;
  using std::fabs;

  int K = 4;
  int K_choose_2 = 6;
  Matrix<var,Dynamic,1> X(K_choose_2);
  X << 1.0, 2.0, -3.0, 1.7, 9.8, -1.2;
  std::vector<var> x;
  for (int i = 0; i < X.size(); ++i)
    x.push_back(X(i));
  var lp = 0.0;
  Matrix<var,Dynamic,Dynamic> Sigma = stan::prob::corr_matrix_constrain(X,K,lp);
  std::vector<var> y;
  for (int m = 0; m < K; ++m)
    for (int n = 0; n < m; ++n)
      y.push_back(Sigma(m,n));
  EXPECT_EQ(K_choose_2, y.size());

  std::vector<std::vector<double> > j;
  stan::agrad::jacobian(y,x,j);

  Matrix<double,Dynamic,Dynamic> J(X.size(),X.size());
  for (int m = 0; m < J.rows(); ++m)
    for (int n = 0; n < J.cols(); ++n)
      J(m,n) = j[m][n];

  double log_abs_jacobian_det = log(fabs(determinant(J)));
  EXPECT_FLOAT_EQ(log_abs_jacobian_det,lp.val());
}

