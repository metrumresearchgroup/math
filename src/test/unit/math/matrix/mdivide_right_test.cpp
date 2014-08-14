#include <stan/math/matrix/mdivide_right.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix,mdivide_right_val) {
  using stan::math::mdivide_right;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_d I;

  Ad << 2.0, 3.0, 
        5.0, 7.0;

  I = mdivide_right(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}

TEST(MathMatrix,mdivide_right_val2) {
  using stan::math::mdivide_right;
  stan::math::row_vector_d b(5);
  stan::math::matrix_d A(5,5);
  stan::math::row_vector_d expected(5);
  stan::math::row_vector_d x;

  b << 19, 150, -170, 140, 31;
  A << 
    1, 8, -9, 7, 5, 
    0, 1, 0, 4, 4, 
    0, 0, 1, 2, 5, 
    0, 0, 0, 1, -5, 
    0, 0, 0, 0, 1;
  expected << 19, -2, 1, 13, 4;
  x = mdivide_right(b, A);
  
  ASSERT_EQ(expected.size(), x.size());
  for (int n = 0; n < expected.size(); n++)
    EXPECT_FLOAT_EQ(expected(n), x(n));
}

TEST(MathMatrix, mdivide_right_nan) {
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
        
  using stan::math::mdivide_right;
    
  expect_matrix_is_nan(mdivide_right(v1, m1));
  expect_matrix_is_nan(mdivide_right(v1, m2));
  expect_matrix_is_nan(mdivide_right(v1, m3));
  expect_matrix_is_nan(mdivide_right(v2, m0));
  expect_matrix_is_nan(mdivide_right(v2, m1));
  expect_matrix_is_nan(mdivide_right(v2, m2));
  expect_matrix_is_nan(mdivide_right(v2, m3));
}
