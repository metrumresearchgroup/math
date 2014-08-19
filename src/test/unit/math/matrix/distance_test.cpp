#include <stan/math/matrix/distance.hpp>
#include <gtest/gtest.h>

#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix, distance_vector_vector) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> v1, v2;
  
  v1.resize(3);
  v2.resize(3);
  v1 << 1, 3, -5;
  v2 << 4, -2, -1;
  
  EXPECT_FLOAT_EQ(7.071068, stan::math::distance(v1, v2));

  v1.resize(0);
  v2.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::distance(v1, v2));

  v1.resize(1);
  v2.resize(2);
  v1 << 1;
  v2 << 2, 3;
  EXPECT_THROW(stan::math::distance(v1, v2), std::domain_error);
}

TEST(MathMatrix, distance_rowvector_vector) {
  Eigen::Matrix<double, 1, Eigen::Dynamic> rv;
  Eigen::Matrix<double, Eigen::Dynamic, 1> v;
  
  rv.resize(3);
  v.resize(3);
  rv << 1, 3, -5;
  v << 4, -2, -1;
  EXPECT_FLOAT_EQ(7.071068, stan::math::distance(rv, v));

  rv.resize(0);
  v.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::distance(rv, v));

  rv.resize(1);
  v.resize(2);
  rv << 1;
  v << 2, 3;
  EXPECT_THROW(stan::math::distance(rv, v), std::domain_error);
}

TEST(MathMatrix, distance_vector_rowvector) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> v;
  Eigen::Matrix<double, 1, Eigen::Dynamic> rv;
  
  v.resize(3);
  rv.resize(3);
  v << 1, 3, -5;
  rv << 4, -2, -1;
  EXPECT_FLOAT_EQ(7.071068, stan::math::distance(v, rv));

  v.resize(0);
  rv.resize(0);
  EXPECT_FLOAT_EQ(0, stan::math::distance(v, rv));

  v.resize(1);
  rv.resize(2);
  v << 1;
  rv << 2, 3;
  EXPECT_THROW(stan::math::distance(v, rv), std::domain_error);
}

TEST(MathMatrix, distance_special_values) {
  Eigen::Matrix<double, Eigen::Dynamic, 1> v1, v2;
  v1.resize(1);
  v2.resize(1);
  
  v1 << 0;
  v2 << std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v2, v1)));

  v1 << 0;
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v2, v1)));

  v1 << std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isnan(stan::math::distance(v2, v1)));

  v1 << -std::numeric_limits<double>::infinity();
  v2 << std::numeric_limits<double>::infinity();
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v1, v2)));
  EXPECT_TRUE(boost::math::isinf(stan::math::distance(v2, v1)));
}

TEST(MathMatrix, distance_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::VectorXd v1(3);
  v1 << 10.2, 1, 1.1;
  
  Eigen::VectorXd v2(3);
  v2 << nan, 6, 5;  
  
  Eigen::RowVectorXd rv1(3);
  rv1 << 0.1, 0.2, nan;
  
  Eigen::RowVectorXd rv2(3);
  rv2 << 10.1, 10,-12;  
        
  using stan::math::distance;
  using boost::math::isnan;
    
  EXPECT_PRED1(isnan<double>, distance(v1, rv1));
  EXPECT_PRED1(isnan<double>, distance(rv1, v1));
  EXPECT_PRED1(isnan<double>, distance(rv2, rv1));
  EXPECT_PRED1(isnan<double>, distance(rv1, rv2));
  EXPECT_PRED1(isnan<double>, distance(v2, v2));
}
