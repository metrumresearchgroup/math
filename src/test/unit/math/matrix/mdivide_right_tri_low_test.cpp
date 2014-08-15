#include <stan/math/matrix/mdivide_right_tri_low.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(MathMatrix, mdivide_right_tri_low_nan) {
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

  m1.transposeInPlace();
  m2.transposeInPlace();
  m3.transposeInPlace();
  m4.transposeInPlace();
  m5.transposeInPlace();
  
  Eigen::RowVectorXd v1(3);
  v1 << 3, 2, 1;
  
  Eigen::RowVectorXd v2(3);
  v2 << 3, nan, 1;
  
  Eigen::RowVectorXd v3(3);
  v3 << 3, 2, nan;
  
  Eigen::RowVectorXd vr;
        
  using stan::math::mdivide_right_tri_low;
    
  vr = mdivide_right_tri_low(v1, m1);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_right_tri_low(v1, m2);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  expect_matrix_not_nan(mdivide_right_tri_low(v1, m3));
  expect_matrix_is_nan(mdivide_right_tri_low(v1, m4));
  
  vr = mdivide_right_tri_low(v1, m5);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_DOUBLE_EQ(vr(1), 0.189);
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_right_tri_low(v2, m0);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_right_tri_low(v2, m1);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_right_tri_low(v2, m2);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  vr = mdivide_right_tri_low(v2, m3);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  expect_matrix_is_nan(mdivide_right_tri_low(v2, m4));

  vr = mdivide_right_tri_low(v2, m5);
  EXPECT_DOUBLE_EQ(vr(2), 0.1);
  EXPECT_PRED1(boost::math::isnan<double>, vr(1));
  EXPECT_PRED1(boost::math::isnan<double>, vr(0));
  
  expect_matrix_is_nan(mdivide_right_tri_low(v3, m0));
  expect_matrix_is_nan(mdivide_right_tri_low(v3, m1));
  expect_matrix_is_nan(mdivide_right_tri_low(v3, m2));
  expect_matrix_is_nan(mdivide_right_tri_low(v3, m3));
  expect_matrix_is_nan(mdivide_right_tri_low(v3, m4));
  expect_matrix_is_nan(mdivide_right_tri_low(v3, m5));
  
}
