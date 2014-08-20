#include <stan/math/matrix/singular_values.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix, singular_values) {
  stan::math::matrix_d m0;

  using stan::math::singular_values;
  EXPECT_NO_THROW(singular_values(m0));
}

TEST(MathMatrix,singular_values_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
        
  Eigen::MatrixXd m1(2,2);
  m1 << 14, nan,
        nan, 10;
        
  Eigen::MatrixXd m2(2,2);
  m2 << nan, 1,
        3, 5.1;
        
  Eigen::MatrixXd m3(2,2);
  m3 << 3, 1,
        3, nan;
  
  Eigen::VectorXd vr;
  
  using stan::math::singular_values;
  using boost::math::isnan;

  expect_matrix_not_nan(singular_values(m1));
  expect_matrix_is_nan(singular_values(m2));
  expect_matrix_is_nan(singular_values(m3));
}
