#include <stan/math/prim/mat/fun/cov_matrix_free.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathPrim, cov_matrix_free_exception) {
  Matrix<double,Dynamic,Dynamic> y(0,0);
  
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
  y.resize(0,10);
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
  y.resize(10,0);
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
  y.resize(1,2);
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);

  y.resize(2,2);
  y << 0, 0, 0, 0;
  EXPECT_THROW(stan::prob::cov_matrix_free(y), std::domain_error);
}
