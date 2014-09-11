#include <stan/math/matrix/qr_Q.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix, qr_Q) {
  stan::math::matrix_d m0(0,0);
  stan::math::matrix_d m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::qr_Q;
  using stan::math::transpose;
  EXPECT_THROW(qr_Q(m0),std::domain_error);
  EXPECT_NO_THROW(qr_Q(m1));
  EXPECT_THROW(qr_Q(transpose(m1)),std::domain_error);
}

TEST(MathMatrix,qr_Q_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
        
  Eigen::MatrixXd m1(2,2);
  m1 << nan, nan,
        nan, nan;
        
  Eigen::MatrixXd m2(2,2);
  m2 << nan, 1,
        3, 5.1;
        
  Eigen::MatrixXd m3(2,2);
  m3 << 3, 1,
        3, nan;
    
  using stan::math::qr_Q;
  using boost::math::isnan;

  expect_matrix_is_nan(qr_Q(m1));
  expect_matrix_is_nan(qr_Q(m2));
  expect_matrix_not_nan(qr_Q(m3));
}
