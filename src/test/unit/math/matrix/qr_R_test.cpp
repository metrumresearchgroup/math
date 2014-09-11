#include <stan/math/matrix/qr_Q.hpp>
#include <stan/math/matrix/qr_R.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix, qr_R) {
  stan::math::matrix_d m0(0,0);
  stan::math::matrix_d m1(3,2);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::qr_R;
  using stan::math::qr_Q;
  using stan::math::transpose;
  EXPECT_THROW(qr_R(m0),std::domain_error);
  EXPECT_NO_THROW(qr_R(m1));

  stan::math::matrix_d m2(3,2);
  m2 = qr_Q(m1) * qr_R(m1);
  for (unsigned int i=0; i<m1.rows(); i++) {
    for (unsigned int j=0; j<m1.cols(); j++) {
      EXPECT_NEAR(m1(i,j), m2(i,j), 1e-8);
    }
  }
  EXPECT_THROW(qr_R(transpose(m1)),std::domain_error);
}

TEST(MathMatrix,qr_R_nan) {
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
  
  Eigen::MatrixXd m3_(2,2);
  m3_ << 3, 1,
         3, 100;
  
  Eigen::MatrixXd mr(2, 2);
  mr << nan, nan,
        0, nan;
  
  using stan::math::qr_R;
  using boost::math::isnan;

  expect_matrix_eq_or_both_nan(qr_R(m1), mr);
  expect_matrix_eq_or_both_nan(qr_R(m2), mr);
  
  mr << qr_R(m3_)(0,0), nan,
        0, nan;
  expect_matrix_eq_or_both_nan(qr_R(m3), mr);
}
