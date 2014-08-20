#include <stan/math/matrix/tcrossprod.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

void test_tcrossprod(const stan::math::matrix_d& x) {
  using stan::math::tcrossprod;
  stan::math::matrix_d y = tcrossprod(x);
  stan::math::matrix_d xxt = x * x.transpose();
  EXPECT_EQ(y.rows(),xxt.rows());
  EXPECT_EQ(y.cols(),xxt.cols());
  for (int m = 0; m < y.rows(); ++m)
    for (int n = 0; n < y.cols(); ++n)
      EXPECT_FLOAT_EQ(xxt(m,n),y(m,n));
}
TEST(MathMatrix, tcrossprod) {
  stan::math::matrix_d x;
  test_tcrossprod(x);

  x = stan::math::matrix_d(1,1);
  x << 3.0;
  test_tcrossprod(x);

  x = stan::math::matrix_d(2,2);
  x <<
    1.0, 0.0,
    2.0, 3.0;
  test_tcrossprod(x);

  x = stan::math::matrix_d(3,3);
  x <<
    1.0, 0.0, 0.0,
    2.0, 3.0, 0.0,
    4.0, 5.0, 6.0;
  test_tcrossprod(x);
}

TEST(MathMatrix,tcrossprod_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,2);
  m1 << 1, nan,
        3, 4.1,
        nan, 6;
  m1.transposeInPlace();
  
  Eigen::MatrixXd m2(3,2);
  m2 << 10.1, 100,
        nan, 0,
        -10, -12.1;
  m2.transposeInPlace();
        
  Eigen::RowVectorXd v1(2);
  v1 << 11.1, nan;
  
  Eigen::MatrixXd mr;
  
  using stan::math::tcrossprod;
  using boost::math::isnan;

  mr = tcrossprod(m1);
  for (int j = 0; j < mr.cols(); j++)
    for (int i = 0; i < mr.rows(); i++)
      EXPECT_PRED1(isnan<double>, mr(i, j));

  mr = tcrossprod(m2);
  EXPECT_PRED1(isnan<double>, mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_PRED1(isnan<double>, mr(1, 0));
  EXPECT_DOUBLE_EQ(10146.41, mr(1, 1)); //100^2+(-12.1)^2

  mr = tcrossprod(v1);
  EXPECT_PRED1(isnan<double>, mr(0, 0));

  mr = tcrossprod(v1.transpose());
  EXPECT_DOUBLE_EQ(123.21, mr(0, 0));
  EXPECT_PRED1(isnan<double>, mr(0, 1));
  EXPECT_PRED1(isnan<double>, mr(1, 0));
  EXPECT_PRED1(isnan<double>, mr(1, 1));
}
