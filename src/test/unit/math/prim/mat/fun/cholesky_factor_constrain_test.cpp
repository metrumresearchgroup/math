#include <stan/math/prim/mat/fun/cholesky_factor_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_free.hpp>
#include <test/unit/util.hpp>

#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathPrim, choleskyFactor) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::prob::cholesky_factor_constrain;
  using stan::prob::cholesky_factor_free;
  
  Matrix<double,Dynamic,1> x(3);
  x << 1, 2, 3;
  
  Matrix<double,Dynamic,Dynamic> y
    = cholesky_factor_constrain(x,2,2);

  Matrix<double,Dynamic,1> x2
    = cholesky_factor_free(y);
  
  EXPECT_EQ(x2.size(), x.size());
  EXPECT_EQ(x2.rows(), x.rows());
  EXPECT_EQ(x2.cols(), x.cols());
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(x(i), x2(i));
}
TEST(MathPrim, choleskyFactorLogJacobian) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::prob::cholesky_factor_constrain;

  double lp;
  Matrix<double,Dynamic,1> x(3);

  x.resize(1);
  x << 2.3;
  lp = 1.9;
  cholesky_factor_constrain(x,1,1,lp);
  EXPECT_FLOAT_EQ(1.9 + 2.3, lp);
  
  x.resize(3);
  x << 
    1, 
    2, 3;
  lp = 7.2;
  cholesky_factor_constrain(x,2,2,lp);
  EXPECT_FLOAT_EQ(7.2 + 1 + 3, lp);

  x.resize(6);
  x << 
    1.001,
    2, 3.01,
    4, 5, 6.1;
  lp = 1.2;
  cholesky_factor_constrain(x,3,3,lp);
  EXPECT_FLOAT_EQ(1.2 + 1.001 + 3.01 + 6.1, lp);

  x.resize(9);
  lp = 1.2;
  x << 
    1.001,
    2, 3.01,
    4, 5, 6.1,
    7, 8, 9;
  cholesky_factor_constrain(x,4,3,lp);
  EXPECT_FLOAT_EQ(1.2 + 1.001 + 3.01 + 6.1, lp);

}
TEST(MathPrim, choleskyFactorConstrainError) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::prob::cholesky_factor_constrain;

  Matrix<double,Dynamic,1> x(3);
  x << 1, 2, 3;
  EXPECT_THROW(cholesky_factor_constrain(x,9,9),std::domain_error);
  double lp = 0;
  EXPECT_THROW(cholesky_factor_constrain(x,9,9,lp),std::domain_error);
}
