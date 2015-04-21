#include <stan/math/prim/mat/fun/unit_vector_free.hpp>
#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;


TEST(MathPrim, unit_vector_free_exception) {
  Matrix<double,Dynamic,1> y(2);
  y << 0.5, 0.55;
  EXPECT_THROW(stan::prob::unit_vector_free(y), std::domain_error);
  y << 1.1, -0.1;
  EXPECT_THROW(stan::prob::unit_vector_free(y), std::domain_error);
}
