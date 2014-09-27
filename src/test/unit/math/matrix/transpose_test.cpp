#include <stan/math/matrix/transpose.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix, transpose) {
  stan::math::vector_d v0;
  stan::math::row_vector_d rv0;
  stan::math::matrix_d m0;

  using stan::math::transpose;
  EXPECT_NO_THROW(transpose(v0));
  EXPECT_NO_THROW(transpose(rv0));
  EXPECT_NO_THROW(transpose(m0));
}

TEST(MathMatrix, transpose_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  
  using stan::math::transpose;
  using boost::math::isnan;

  Eigen::MatrixXd m1(3,3);
  m1 << 10, 1.5, 3.2,
        nan, 10, 4.1,
        0.1, 4.1, 10;

  Eigen::MatrixXd mr = transpose(m1);
  
  EXPECT_EQ(10, mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_EQ(0.1, mr(0, 2));
  
  EXPECT_EQ(1.5, mr(1, 0));
  EXPECT_EQ(10, mr(1, 1));
  EXPECT_EQ(4.1, mr(1, 2));
  
  EXPECT_EQ(3.2, mr(2, 0));
  EXPECT_EQ(4.1, mr(2, 1));
  EXPECT_EQ(10, mr(2, 2));

  Eigen::VectorXd v1(4);
  v1 << 10, 3.2, nan, 5.1;

  Eigen::RowVectorXd rvr = transpose(v1);
  mr = transpose(v1);

  EXPECT_EQ(10, rvr(0));
  EXPECT_EQ(3.2, rvr(1));
  EXPECT_PRED1(isnan<double>, rvr(2));
  EXPECT_EQ(5.1, rvr(3));
  
  EXPECT_EQ(10, mr(0, 0));
  EXPECT_EQ(3.2, mr(0, 1));
  EXPECT_PRED1(isnan<double>, mr(0, 2));
  EXPECT_EQ(5.1, mr(0, 3));
    
  Eigen::VectorXd rv1(4);
  rv1 << 10, 3.2, nan, 5.1;

  Eigen::RowVectorXd vr = transpose(rv1);
  mr = transpose(rv1);

  EXPECT_EQ(10, vr(0));
  EXPECT_EQ(3.2, vr(1));
  EXPECT_PRED1(isnan<double>, vr(2));
  EXPECT_EQ(5.1, vr(3));
  
  EXPECT_EQ(10, mr(0, 0));
  EXPECT_EQ(3.2, mr(0, 1));
  EXPECT_PRED1(isnan<double>, mr(0, 2));
  EXPECT_EQ(5.1, mr(0, 3));
}

