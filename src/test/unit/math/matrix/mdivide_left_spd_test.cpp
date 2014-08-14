#include <stan/math/matrix/mdivide_left_spd.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix,mdivide_left_spd_val) {
  using stan::math::mdivide_left_spd;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_d I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;

  I = mdivide_left_spd(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}

TEST(MathMatrix, mdivide_left_spd_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m0(3,3);
  m0 << 10, 1, 3.2,
        1.1, 10, 4.1,
        0.1, 4.1, 10;

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
  
  Eigen::VectorXd v1(3);
  v1 << 1, 2, 3;
  
  Eigen::VectorXd v2(3);
  v2 << 1, nan, 3;
        
  using stan::math::mdivide_left_spd;
    
  expect_matrix_nan(mdivide_left_spd(m1, v1));
  expect_matrix_nan(mdivide_left_spd(m2, v1));
  expect_matrix_not_nan(mdivide_left_spd(m3, v1));
  expect_matrix_nan(mdivide_left_spd(m0, v2));
  expect_matrix_nan(mdivide_left_spd(m1, v2));
  expect_matrix_nan(mdivide_left_spd(m2, v2));
  expect_matrix_nan(mdivide_left_spd(m3, v2));
}
