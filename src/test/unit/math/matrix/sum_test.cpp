#include <stan/math/matrix/sum.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix,sum_vector_int) {
  std::vector<int> x(3);
  EXPECT_EQ(0,stan::math::sum(x));
  x[0] = 1;
  x[1] = 2;
  x[2] = 3;
  EXPECT_EQ(6,stan::math::sum(x));
}
TEST(MathMatrix,sum_vector_double) {
  using stan::math::sum;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  std::vector<double> x(3);
  EXPECT_FLOAT_EQ(0.0,sum(x));
  x[0] = 1.0;
  x[1] = 2.0;
  x[2] = 3.0;
  EXPECT_FLOAT_EQ(6.0,sum(x));

  stan::math::vector_d v;
  EXPECT_FLOAT_EQ(0.0,sum(v));
  v = stan::math::vector_d(1);
  v[0] = 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(v));
  v = stan::math::vector_d(3);
  v[0] = 5.0;
  v[1] = 10.0;
  v[2] = 100.0;
  EXPECT_FLOAT_EQ(115.0,sum(v));

  stan::math::row_vector_d rv;
  EXPECT_FLOAT_EQ(0.0,sum(rv));
  rv = stan::math::row_vector_d(1);
  rv[0] = 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(rv));
  rv = stan::math::row_vector_d(3);
  rv[0] = 5.0;
  rv[1] = 10.0;
  rv[2] = 100.0;
  EXPECT_FLOAT_EQ(115.0,sum(rv));

  stan::math::matrix_d m;
  EXPECT_FLOAT_EQ(0.0,sum(m));
  m = stan::math::matrix_d(1,1);
  m << 5.0;
  EXPECT_FLOAT_EQ(5.0,sum(m));
  m = stan::math::matrix_d(3,2);
  m << 1, 2, 3, 4, 5, 6;
  EXPECT_FLOAT_EQ(21.0,sum(m));
}

TEST(MathMatrix, sum_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,2);
  m1 << 1, nan,
        3, 4.1,
        nan, 6;
  Eigen::MatrixXd m2(3,2);
  m2 << 10.1, 100,
        nan, 0,
        -10, -12;
        
  Eigen::VectorXd v1(3);
  v1 << 10.1, nan, 1.1;
        
  std::vector<double> v2(14, 1.1);
  v2[7] = nan;
        
  using stan::math::sum;
  using boost::math::isnan;
    
  EXPECT_PRED1(isnan<double>, sum(m1));
  EXPECT_PRED1(isnan<double>, sum(m2));
  EXPECT_PRED1(isnan<double>, sum(v1));
  EXPECT_PRED1(isnan<double>, sum(v2));
}
