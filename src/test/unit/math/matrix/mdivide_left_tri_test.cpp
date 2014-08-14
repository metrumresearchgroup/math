#include <stan/math/matrix/mdivide_left_tri.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix,mdivide_left_tri_val) {
  using stan::math::mdivide_left_tri;
  stan::math::matrix_d Ad(2,2);
  stan::math::matrix_d Ad_inv(2,2);
  stan::math::matrix_d I;

  Ad << 2.0, 0.0, 
        5.0, 7.0;

  I = mdivide_left_tri<Eigen::Lower>(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);

  Ad_inv = mdivide_left_tri<Eigen::Lower>(Ad);
  I = Ad * Ad_inv;
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);

  Ad << 2.0, 3.0, 
        0.0, 7.0;

  I = mdivide_left_tri<Eigen::Upper>(Ad,Ad);
  EXPECT_NEAR(1.0,I(0,0),1.0E-12);
  EXPECT_NEAR(0.0,I(0,1),1.0E-12);
  EXPECT_NEAR(0.0,I(1,0),1.0E-12);
  EXPECT_NEAR(1.0,I(1,1),1.0e-12);
}

TEST(MathMatrix, mdivide_left_tri_lower_nan) {
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
          
  Eigen::MatrixXd m4(3,3);
  m4 << nan, 1, 3.2,
        1.1, 10, 4.1,
        0.1, 4.1, 10;
          
  Eigen::MatrixXd m5(3,3);
  m5 << 10, 1, 3.2,
        1.1, 10, 4.1,
        0.1, nan, 10;
  
  Eigen::VectorXd v1(3);
  v1 << 1, 2, 3;
  
  Eigen::VectorXd v2(3);
  v2 << 1, nan, 3;
  
  Eigen::VectorXd v3(3);
  v3 << nan, 2, 3;
  
  Eigen::VectorXd vr;
        
  using stan::math::mdivide_left_tri;
    
  vr = mdivide_left_tri<Eigen::Lower>(m1, v1);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  vr = mdivide_left_tri<Eigen::Lower>(m2, v1);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  expect_matrix_not_nan(mdivide_left_tri<Eigen::Lower>(m3, v1));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Lower>(m4, v1));
  
  vr = mdivide_left_tri<Eigen::Lower>(m5, v1);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  EXPECT_DOUBLE_EQ(vr(1), 0.189);
  EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  vr = mdivide_left_tri<Eigen::Lower>(m0, v2);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  vr = mdivide_left_tri<Eigen::Lower>(m1, v2);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  vr = mdivide_left_tri<Eigen::Lower>(m2, v2);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  vr = mdivide_left_tri<Eigen::Lower>(m3, v2);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Lower>(m4, v2));

  vr = mdivide_left_tri<Eigen::Lower>(m5, v2);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Lower>(m0, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Lower>(m1, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Lower>(m2, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Lower>(m3, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Lower>(m4, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Lower>(m5, v3));
  
  //1 arg function tests begin
  vr = mdivide_left_tri<Eigen::Lower>(m1);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  //EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  //EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  vr = mdivide_left_tri<Eigen::Lower>(m2);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  //EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  //EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
  expect_matrix_not_nan(mdivide_left_tri<Eigen::Lower>(m3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Lower>(m4));

  vr = mdivide_left_tri<Eigen::Lower>(m5);
  EXPECT_DOUBLE_EQ(vr(0), 0.1);
  //EXPECT_DOUBLE_EQ(vr(1), 0.089);
  //EXPECT_PRED1(boost::math::isnan<double>, vr(2));
  
}



TEST(MathMatrix, mdivide_left_tri_upper_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m0(3,3);
  m0 << 10, 4.1, 0.1,
        4.1, 10, 1.1,
        3.2, 1, 10;

  Eigen::MatrixXd m1(3,3);
  m1 << 10, 4.1, 0.1,
        4.1, 10, nan,
        3.2, 1, 10;
        
  Eigen::MatrixXd m2(3,3);
  m2 << 10, 4.1, 0.1,
        4.1, nan, 1.1,
        3.2, 1, 10;
          
  Eigen::MatrixXd m3(3,3);
  m3 << 10, 4.1, 0.1,
        4.1, 10, 1.1,
        nan, 1, 10;
          
  Eigen::MatrixXd m4(3,3);
  m4 << 10, 4.1, 0.1,
        4.1, 10, 1.1,
        3.2, 1, nan;
          
  Eigen::MatrixXd m5(3,3);
  m5 << 10, nan, 0.1,
        4.1, 10, 1.1,
        3.2, 1, 10;
  
  Eigen::VectorXd v1(3);
  v1 << 3, 2, 1;
  
  Eigen::VectorXd v2(3);
  v2 << 3, nan, 1;
  
  Eigen::VectorXd v3(3);
  v3 << 3, 2, nan;
  
  Eigen::VectorXd vr;
        
  using stan::math::mdivide_left_tri;
    
  vr = mdivide_left_tri<Eigen::Upper>(m1, v1);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_left_tri<Eigen::Upper>(m2, v1);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  expect_matrix_not_nan(mdivide_left_tri<Eigen::Upper>(m3, v1));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Upper>(m4, v1));
  
  vr = mdivide_left_tri<Eigen::Upper>(m5, v1);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_DOUBLE_EQ(vr(1), 0.189);
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_left_tri<Eigen::Upper>(m0, v2);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_left_tri<Eigen::Upper>(m1, v2);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_left_tri<Eigen::Upper>(m2, v2);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_left_tri<Eigen::Upper>(m3, v2);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Upper>(m4, v2));

  vr = mdivide_left_tri<Eigen::Upper>(m5, v2);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Upper>(m0, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Upper>(m1, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Upper>(m2, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Upper>(m3, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Upper>(m4, v3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Upper>(m5, v3));
  
  //1 arg function tests begin
  vr = mdivide_left_tri<Eigen::Upper>(m1);
  //EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_left_tri<Eigen::Upper>(m2);
  //EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  expect_matrix_not_nan(mdivide_left_tri<Eigen::Upper>(m3));
  expect_matrix_is_nan(mdivide_left_tri<Eigen::Upper>(m4));

  vr = mdivide_left_tri<Eigen::Upper>(m5);
  //EXPECT_DOUBLE_EQ(vr(2), 0.1);
  //EXPECT_DOUBLE_EQ(vr(1), 0.089);
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
}

