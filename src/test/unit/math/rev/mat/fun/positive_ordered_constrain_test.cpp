#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/prim/mat/fun/determinant.hpp>
#include <stan/math/prim/mat/fun/positive_ordered_constrain.hpp>
#include <test/unit/math/rev/mat/fun/jacobian.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathRev, positive_ordered_jacobian_ad) {
  using stan::agrad::var;
  using stan::prob::positive_ordered_constrain;
  using stan::math::determinant;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  Matrix<double,Dynamic,1> x(3);
  x << -12.0, 3.0, -1.9;
  double lp = 0.0;
  Matrix<double,Dynamic,1> y = positive_ordered_constrain(x,lp);

  Matrix<var,Dynamic,1> xv(3);
  xv << -12.0, 3.0, -1.9;

  std::vector<var> xvec(3);
  for (int i = 0; i < 3; ++i)
    xvec[i] = xv[i];

  Matrix<var,Dynamic,1> yv = positive_ordered_constrain(xv);


  EXPECT_EQ(y.size(), yv.size());
  for (int i = 0; i < y.size(); ++i)
    EXPECT_FLOAT_EQ(y(i),yv(i).val());

  std::vector<var> yvec(3);
  for (unsigned int i = 0; i < 3; ++i)
    yvec[i] = yv[i];

  std::vector<std::vector<double> > j;
  stan::agrad::jacobian(yvec,xvec,j);

  Matrix<double,Dynamic,Dynamic> J(3,3);
  for (int m = 0; m < 3; ++m)
    for (int n = 0; n < 3; ++n)
      J(m,n) = j[m][n];
  
  double log_abs_jacobian_det = log(fabs(determinant(J)));
  EXPECT_FLOAT_EQ(log_abs_jacobian_det, lp);
}

