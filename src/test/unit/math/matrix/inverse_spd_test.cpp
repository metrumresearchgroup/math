#include <stan/math/matrix/inverse_spd.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, inverse_spd_exception) {
  using stan::math::inverse_spd;

  stan::math::matrix_d m1(2,3);
  
  // non-square
  m1 << 1, 2, 3, 4, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  stan::math::matrix_d m2(3,3);
  
  // non-symmetric
  m2 << 1, 2, 3, 4, 5, 6, 7, 8, 9;  
  EXPECT_THROW(inverse_spd(m1),std::domain_error);

  // not positive definite
  m2 << 1, 2, 3,
        2, 4, 5,
        3, 5, 6;
  EXPECT_THROW(inverse_spd(m1),std::domain_error);
}

TEST(MathMatrix, inverse_spd_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m0(3,3);
  m0 << 100, 3.2, 0,
        3.2, 100, nan,
        0, nan, 100;

  Eigen::MatrixXd m1(3,3);
  m1 << 100, nan, 0.1,
        nan, 100, 4.1,
        0.1, 4.1, 100;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 100, 1.1, 3.2,
        1.1, nan, 4.1,
        3.2, 4.1, 100;
          
  Eigen::MatrixXd m3(3,3);
  m3 << 100, 1.1, nan,
        1.1, 100, 4.1,
        nan, 4.1, 100;
        
  using stan::math::inverse_spd;
    
  EXPECT_THROW(inverse_spd(m0),std::domain_error);
  EXPECT_THROW(inverse_spd(m1),std::domain_error);
  EXPECT_THROW(inverse_spd(m2),std::domain_error);
  EXPECT_THROW(inverse_spd(m3),std::domain_error);
}
