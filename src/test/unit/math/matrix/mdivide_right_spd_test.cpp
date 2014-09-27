#include <stan/math/matrix/mdivide_right_spd.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix,mdivide_right_spd_val) {
  using stan::math::mdivide_right_spd;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_d I;

  Ad << 2.0, 3.0, 
        3.0, 7.0;

  I = mdivide_right_spd(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}

TEST(MathMatrix, mdivide_right_spd_nan) {
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
  
  Eigen::RowVectorXd v1(3);
  v1 << 1, 2, 3;
  
  Eigen::RowVectorXd v2(3);
  v2 << 1, nan, 3;
        
  using stan::math::mdivide_right_spd;
    
  EXPECT_THROW(mdivide_right_spd(v1, m1), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(v1, m2), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(v1, m3), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(v2, m0), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(v2, m1), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(v2, m2), std::domain_error);
  EXPECT_THROW(mdivide_right_spd(v2, m3), std::domain_error);
}
