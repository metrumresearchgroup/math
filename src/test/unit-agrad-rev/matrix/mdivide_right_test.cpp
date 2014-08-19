#include <stan/math/matrix/mdivide_right.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev.hpp>
#include <test/unit-agrad-rev/matrix/expect_matrix_nan.hpp>

TEST(AgradRevMatrix,mdivide_right_val) {
  using stan::math::matrix_d;
  using stan::agrad::matrix_v;
  using stan::math::mdivide_right;

  matrix_v Av(2,2);
  matrix_d Ad(2,2);
  matrix_v I;

  Av << 2.0, 3.0, 
    5.0, 7.0;
  Ad << 2.0, 3.0, 
    5.0, 7.0;

  I = mdivide_right(Av,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_right(Av,Ad);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);

  I = mdivide_right(Ad,Av);
  EXPECT_NEAR(1.0,I(0,0).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1).val(),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0).val(),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1).val(),1.0e-12);
}

TEST(AgradRevMatrix, mdivide_right_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  stan::agrad::matrix_v m0(3,3);
  m0 << 10, 1, 3.2,
        1.1, 10, 4.1,
        0.1, 4.1, 10;

  stan::agrad::matrix_v m1(3,3);
  m1 << 10, 1, 3.2,
        nan, 10, 4.1,
        0.1, 4.1, 10;
        
  stan::agrad::matrix_v m2(3,3);
  m2 << 10, 1, 3.2,
        1.1, nan, 4.1,
        0.1, 4.1, 10;
          
  stan::agrad::matrix_v m3(3,3);
  m3 << 10, 1, nan,
        1.1, 10, 4.1,
        0.1, 4.1, 10;
  
  stan::agrad::row_vector_v v1(3);
  v1 << 1, 2, 3;
  
  stan::agrad::row_vector_v v2(3);
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
