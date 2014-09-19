#include <stan/math/matrix/softmax.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix,softmax) {
  using stan::math::softmax;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  
  Matrix<double,Dynamic,1> x(1);
  x << 0.0;
  
  Matrix<double,Dynamic,1> theta = softmax(x);
  EXPECT_EQ(1,theta.size());
  EXPECT_FLOAT_EQ(1.0,theta[0]);

  Matrix<double,Dynamic,1> x2(2);
  x2 << -1.0, 1.0;
  Matrix<double,Dynamic,1> theta2 = softmax(x2);
  EXPECT_EQ(2,theta2.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1)), theta2[0]);
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1)), theta2[1]);

  Matrix<double,Dynamic,1> x3(3);
  x3 << -1.0, 1.0, 10.0;
  Matrix<double,Dynamic,1> theta3 = softmax(x3);
  EXPECT_EQ(3,theta3.size());
  EXPECT_FLOAT_EQ(exp(-1)/(exp(-1) + exp(1) + exp(10.0)), theta3[0]);
  EXPECT_FLOAT_EQ(exp(1)/(exp(-1) + exp(1) + exp(10.0)), theta3[1]);
  EXPECT_FLOAT_EQ(exp(10)/(exp(-1) + exp(1) + exp(10.0)), theta3[2]);
}
TEST(MathMatrix,softmax_exception) {
  using stan::math::softmax;
  stan::math::vector_d v0;

  EXPECT_THROW(softmax(v0),std::domain_error);
}  

TEST(MathMatrix, softmax_nan) {
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
        
  using stan::math::softmax;
    
  expect_matrix_is_nan(softmax(m0));
  expect_matrix_is_nan(softmax(m1));
  expect_matrix_is_nan(softmax(m2));
  expect_matrix_is_nan(softmax(m3));
}
