#include <stan/math/matrix/cholesky_decompose.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <gtest/gtest.h>

TEST(MathMatrix, cholesky_decompose) {
  stan::math::matrix_d m0;
  stan::math::matrix_d m1(2,3);
  m1 << 1, 2, 3, 4, 5, 6;

  using stan::math::cholesky_decompose;

  EXPECT_NO_THROW(cholesky_decompose(m0));
  EXPECT_THROW(cholesky_decompose(m1),std::domain_error);
}

TEST(MathMatrix, cholesky_decompose_exception) {
  stan::math::matrix_d m;
  
  m.resize(2,2);
  m << 1.0, 2.0, 
    2.0, 3.0;
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));

  m.resize(0, 0);
  EXPECT_NO_THROW(stan::math::cholesky_decompose(m));
  
  m.resize(2, 3);
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);

  // not symmetric
  m.resize(2,2);
  m << 1.0, 2.0,
    3.0, 4.0;
  EXPECT_THROW(stan::math::cholesky_decompose(m), std::domain_error);
}

TEST(MathMatrix,cholesky_decompose_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
        
  Eigen::MatrixXd m1(2,2);
  m1 << 1, 2,
        2, 13.61;
        
  Eigen::MatrixXd m2(2,2);
  m2 << nan, 2,
        2, 13.61;
        
  Eigen::MatrixXd m3(2,2);
  m3 << 1, nan,
        nan, 13.61;
        
  Eigen::MatrixXd m4(2,2);
  m4 << 1, 2,
        2, nan;
        
  Eigen::MatrixXd m5(2,2);
  m5 << nan, nan,
        nan, nan;
  
  Eigen::MatrixXd mr;
  
  using stan::math::cholesky_decompose;
  using boost::math::isnan;
  
  mr = cholesky_decompose(m1);
  EXPECT_DOUBLE_EQ(mr(0), 1);
  EXPECT_DOUBLE_EQ(mr(1), 2);
  EXPECT_DOUBLE_EQ(mr(2), 0);
  EXPECT_DOUBLE_EQ(mr(3), 3.1);
  
  EXPECT_THROW(cholesky_decompose(m2), std::domain_error);
  EXPECT_THROW(mr = cholesky_decompose(m3), std::domain_error);
  EXPECT_THROW(mr = cholesky_decompose(m4), std::domain_error);
  EXPECT_THROW(mr = cholesky_decompose(m5), std::domain_error);
}
