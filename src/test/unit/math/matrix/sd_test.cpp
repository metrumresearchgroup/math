#include <stan/math/matrix/sd.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix,sd) {
  using stan::math::sd;
  std::vector<double> x;
  EXPECT_THROW(sd(x),std::domain_error);
  x.push_back(1.0);
  EXPECT_FLOAT_EQ(0.0,sd(x));
  x.push_back(2.0);
  EXPECT_NEAR(0.7071068,sd(x),0.000001);
  x.push_back(3.0);
  EXPECT_FLOAT_EQ(1.0,sd(x));

  stan::math::vector_d v;
  EXPECT_THROW(sd(v),std::domain_error);
  v = stan::math::vector_d(1);
  v << 1.0;
  EXPECT_FLOAT_EQ(0.0,sd(v));
  v = stan::math::vector_d(2);
  v << 1.0, 2.0;
  EXPECT_NEAR(0.7071068,sd(v),0.000001);
  v = stan::math::vector_d(3);
  v << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,sd(v));

  stan::math::row_vector_d rv;
  EXPECT_THROW(sd(rv),std::domain_error);
  rv = stan::math::row_vector_d(1);
  rv << 1.0;
  EXPECT_FLOAT_EQ(0.0,sd(rv));
  rv = stan::math::row_vector_d(2);
  rv << 1.0, 2.0;
  EXPECT_NEAR(0.7071068,sd(rv),0.000001);
  rv = stan::math::row_vector_d(3);
  rv << 1.0, 2.0, 3.0;
  EXPECT_FLOAT_EQ(1.0,sd(rv));


  stan::math::matrix_d m;
  EXPECT_THROW(sd(m),std::domain_error);
  m = stan::math::matrix_d(1,1);
  m << 1.0;
  EXPECT_FLOAT_EQ(0.0,sd(m));
  m = stan::math::matrix_d(2,3);
  m << 1.0, 2.0, 4.0, 9.0, 16.0, 25.0;
  EXPECT_NEAR(9.396808,sd(m),0.000001);
}

TEST(MathMatrix, sd_nan) {
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
        
  using stan::math::sd;
  using boost::math::isnan;
    
  EXPECT_PRED1(isnan<double>, sd(m1));
  EXPECT_PRED1(isnan<double>, sd(m2));
  EXPECT_PRED1(isnan<double>, sd(v1));
  EXPECT_PRED1(isnan<double>, sd(v2));
}
