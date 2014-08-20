#include <stan/math/matrix/inverse.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix, inverse_exception) {
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::inverse;
  EXPECT_THROW(inverse(m1),std::domain_error);
}

TEST(MathMatrix, inverse_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m0(3,3);
  m0 << 0, 1, 3.2,
        0, 10, 4.1,
        0, nan, 10;

  Eigen::MatrixXd m1(3,3);
  m1 << 10, 1, 3.2,
        nan, 10, 4.1,
        0.1, 4.1, 10;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 10, 1, 3.2,
        1.1, nan, 4.1,
        0.1, 4.1, 10;
          
  Eigen::MatrixXd m3(3,3);
  m3 << 10, 1, nan,
        1.1, 10, 4.1,
        0.1, 4.1, 10;
        
  using stan::math::inverse;
    
  expect_matrix_is_nan(inverse(m0));
  expect_matrix_is_nan(inverse(m1));
  expect_matrix_is_nan(inverse(m2));
  expect_matrix_is_nan(inverse(m3));
}
