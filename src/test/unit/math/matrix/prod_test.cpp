#include <stan/math/matrix/prod.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix,prod_vector_int) {
  using stan::math::prod;
  std::vector<int> v;
  EXPECT_EQ(1,prod(v));
  v.push_back(2);
  EXPECT_EQ(2,prod(v));
  v.push_back(3);
  EXPECT_EQ(6,prod(v));
}
TEST(MathMatrix,prod_vector_double) {
  using stan::math::prod;
  std::vector<double> x;
  EXPECT_FLOAT_EQ(1.0,prod(x));
  x.push_back(2.0);
  EXPECT_FLOAT_EQ(2.0,prod(x));
  x.push_back(3);
  EXPECT_FLOAT_EQ(6.0,prod(x));

  stan::math::vector_d v;
  EXPECT_FLOAT_EQ(1.0,prod(v));
  v = stan::math::vector_d(1);
  v << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(v));
  v = stan::math::vector_d(2);
  v << 2.0, 3.0;
  EXPECT_FLOAT_EQ(6.0,prod(v));

  stan::math::row_vector_d rv;
  EXPECT_FLOAT_EQ(1.0,prod(rv));
  rv = stan::math::row_vector_d(1);
  rv << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(rv));
  rv = stan::math::row_vector_d(2);
  rv << 2.0, 3.0;
  EXPECT_FLOAT_EQ(6.0,prod(rv));

  stan::math::matrix_d m;
  EXPECT_FLOAT_EQ(1.0,prod(m));
  m = stan::math::matrix_d(1,1);
  m << 2.0;
  EXPECT_FLOAT_EQ(2.0,prod(m));
  m = stan::math::matrix_d(2,3);
  m << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0;
  EXPECT_FLOAT_EQ(720.0,prod(m));
}

TEST(MathMatrix, prod_nan) {
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
        
  using stan::math::prod;
  using boost::math::isnan;
    
  EXPECT_PRED1(isnan<double>, prod(m1));
  EXPECT_PRED1(isnan<double>, prod(m2));
  EXPECT_PRED1(isnan<double>, prod(v1));
  EXPECT_PRED1(isnan<double>, prod(v2));
}
