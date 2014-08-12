#include <stan/math/matrix/dot_self.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

TEST(MathMatrix, dot_self) {
  using stan::math::dot_self;

  Eigen::Matrix<double,Eigen::Dynamic,1> v1(1);
  v1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(v1),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,1> v2(2);
  v2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(v2),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,1> v3(3);
  v3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(v3),1E-12);

  Eigen::Matrix<double,1,Eigen::Dynamic> rv1(1);
  rv1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(rv1),1E-12);
  Eigen::Matrix<double,1,Eigen::Dynamic> rv2(2);
  rv2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(rv2),1E-12);
  Eigen::Matrix<double,1,Eigen::Dynamic> rv3(3);
  rv3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(rv3),1E-12);

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m1(1,1);
  m1 << 2.0;
  EXPECT_NEAR(4.0,dot_self(m1),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m2(2,1);
  m2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(m2),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> m3(3,1);
  m3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(m3),1E-12);

  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> mm2(1,2);
  mm2 << 2.0, 3.0;
  EXPECT_NEAR(13.0,dot_self(mm2),1E-12);
  Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic> mm3(1,3);
  mm3 << 2.0, 3.0, 4.0;
  EXPECT_NEAR(29.0,dot_self(mm3),1E-12);
}

TEST(MathMatrix, dot_self_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(6,1);
  m1 << 1, nan, 3, 4.1, nan, 6;
  
  Eigen::MatrixXd m2(1,6);
  m2 << 10.1, 100, nan, 0, -10, -12;
        
  Eigen::VectorXd v1(3);
  v1 << 10.1, nan, 1.1;
        
  Eigen::RowVectorXd rv1(3);
  rv1 << 1.1, nan, 2.1;
        
  using stan::math::dot_self;
  using boost::math::isnan;
    
  EXPECT_PRED1(isnan<double>, dot_self(m1));
  EXPECT_PRED1(isnan<double>, dot_self(m2));
  EXPECT_PRED1(isnan<double>, dot_self(v1));
  EXPECT_PRED1(isnan<double>, dot_self(rv1));
}
