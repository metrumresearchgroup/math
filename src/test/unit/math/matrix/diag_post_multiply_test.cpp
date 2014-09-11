#include <stan/math/matrix/diag_post_multiply.hpp>
#include <test/unit/math/matrix/expect_matrix_eq.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::diag_post_multiply;

TEST(MathMatrix,diagPostMultiply) {

  Matrix<double,Dynamic,Dynamic> m(1,1);
  m << 3;

  Matrix<double,Dynamic,1> v(1);
  v << 9;

  Matrix<double,Dynamic,Dynamic> v_m(1,1);
  v_m << 9;
  
  expect_matrix_eq(m * v_m, diag_post_multiply(m,v));
}
TEST(MathMatrix,diagPostMultiply2) {
  Matrix<double,Dynamic,Dynamic> m(2,2);
  m << 
    2, 3,
    4, 5;

  Matrix<double,Dynamic,1> v(2);
  v << 10, 100;

  Matrix<double,Dynamic,Dynamic> v_m(2,2);
  v_m << 
    10, 0,
    0, 100;

  expect_matrix_eq(m * v_m, diag_post_multiply(m,v));

  Matrix<double,1,Dynamic> rv(2);
  rv << 10, 100;
  expect_matrix_eq(m * v_m, diag_post_multiply(m,rv));

}

TEST(MathMatrix,diagPostMultiplyException) {
  Matrix<double,Dynamic,Dynamic> m(2,2);
  m << 
    2, 3,
    4, 5;
  EXPECT_THROW(diag_post_multiply(m,m), std::domain_error);

  Matrix<double,Dynamic,1> v(3);
  v << 1, 2, 3;
  EXPECT_THROW(diag_post_multiply(m,v), std::domain_error);
}

TEST(MathMatrix, diag_post_multiply_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m(3,3);
  m << 1, nan, 1.1,
       5.1, 4.1, 4.5,
       nan, 6.1, 5.1;
        
  Eigen::VectorXd vm(3);
  vm << 10.1, 6, nan;
  
  Eigen::MatrixXd mr(3,3);
  mr << 10.1, nan, nan,
        51.51, 24.6, nan,
        nan, 36.6, nan;  
         
  using stan::math::diag_post_multiply;
  using boost::math::isnan;
  
  expect_matrix_eq_or_both_nan(mr, diag_post_multiply(m, vm));
}
