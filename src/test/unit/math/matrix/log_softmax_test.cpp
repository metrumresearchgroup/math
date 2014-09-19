#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/log_softmax.hpp>
#include <stan/math/matrix/softmax.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>


void test_log_softmax(const Eigen::Matrix<double,Eigen::Dynamic,1>& theta) {
  using std::log;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::log_softmax;
  using stan::math::softmax;

  int size = theta.size();

  Matrix<double,Dynamic,1> log_softmax_theta 
    = log_softmax(theta);

  Matrix<double,Dynamic,1> softmax_theta 
    = softmax(theta);

  Matrix<double,Dynamic,1> log_softmax_theta_expected(size);
  for (int i = 0; i < size; ++i)
    log_softmax_theta_expected(i) = log(softmax_theta(i));
    
  EXPECT_EQ(log_softmax_theta_expected.size(),
            log_softmax_theta.size());
  for (int i = 0; i < theta.size(); ++i)
    EXPECT_FLOAT_EQ(log_softmax_theta_expected(i),
                    log_softmax_theta(i));
}

TEST(MathMatrix,softmax) {
  using stan::math::softmax;
  using stan::math::log_softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  // stan::math::vector_d x(1);
  // x << 0.0;
  // test_log_softmax(x);

  stan::math::vector_d x2(2);
  x2 << -1.0, 1.0;
  test_log_softmax(x2);

  // stan::math::vector_d x3(3);
  // x3 << -1.0, 1.0, 10.0;
  // test_log_softmax(x3);
}
TEST(MathMatrix,softmax_exception) {
  using stan::math::log_softmax;
  stan::math::vector_d v0;  // size == 0

  EXPECT_THROW(log_softmax(v0),std::domain_error);
}  

TEST(MathMatrix, log_softmax_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::VectorXd m0(9);
  m0 << 0, 1, 3.2,
        0, 10, 4.1,
        0, nan, 10;

  Eigen::VectorXd m1(9);
  m1 << 10, 1, 3.2,
        nan, 10, 4.1,
        0.1, 4.1, 10;
        
  Eigen::VectorXd m2(9);
  m2 << 10, 1, 3.2,
        1.1, nan, 4.1,
        0.1, 4.1, 10;
          
  Eigen::VectorXd m3(9);
  m3 << 10, 1, nan,
        1.1, 10, 4.1,
        0.1, 4.1, 10;
        
  using stan::math::log_softmax;
    
  expect_matrix_is_nan(log_softmax(m0));
  expect_matrix_is_nan(log_softmax(m1));
  expect_matrix_is_nan(log_softmax(m2));
  expect_matrix_is_nan(log_softmax(m3));
}
