#include <stan/math/matrix/eigenvalues_sym.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix, eigenvalues_sym) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::eigenvalues_sym;
  EXPECT_THROW(eigenvalues_sym(m0),std::domain_error);
  EXPECT_THROW(eigenvalues_sym(m1),std::domain_error);
}

TEST(MathMatrix,eigenvalues_sym_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,3);
  m1 << 14, nan, 3,
        nan, 10, 4.1,
        3, 4.1, 8;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 10.1, 1, 1.3,
        1, 5.4, 1.5,
        1.3, 1.5, nan;
  
  Eigen::VectorXd vr;
  
  using stan::math::eigenvalues_sym;
  using boost::math::isnan;

  EXPECT_THROW(vr = eigenvalues_sym(m1), std::domain_error);
  EXPECT_THROW(vr = eigenvalues_sym(m2), std::domain_error);
}
