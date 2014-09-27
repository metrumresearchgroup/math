#include <stan/math/matrix/eigenvectors_sym.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix, eigenvectors_sym) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;
  stan::math::matrix_d ev_m1(1,1);
  ev_m1 << 2.0;

  using stan::math::eigenvectors_sym;
  EXPECT_THROW(eigenvectors_sym(m0),std::domain_error);
  EXPECT_NO_THROW(eigenvectors_sym(ev_m1));
  EXPECT_THROW(eigenvectors_sym(m1),std::domain_error);
}

TEST(MathMatrix,eigenvectors_sym_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,3);
  m1 << 14, nan, 3,
        nan, 10, 4.1,
        3, 4.1, 8;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 10.1, 1, 1.3,
        1, 5.4, 1.5,
        1.3, 1.5, nan;
  
  Eigen::MatrixXd mr;
  
  using stan::math::eigenvectors_sym;
  using boost::math::isnan;

  EXPECT_THROW(mr = eigenvectors_sym(m1), std::domain_error);
  EXPECT_THROW(mr = eigenvectors_sym(m2), std::domain_error);
}
