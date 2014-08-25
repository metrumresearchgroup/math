#include <gtest/gtest.h>
#include <stan/math/matrix/fill.hpp>
#include <test/unit/math/matrix/expect_matrix_nan.hpp>

TEST(AgradRevMatrix, fill) {
  using stan::math::fill;
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  double x;
  double y = 10;
  fill(x,y);
  EXPECT_FLOAT_EQ(10.0, x);

  std::vector<double> z(2);
  double a = 15;
  fill(z,a);
  EXPECT_FLOAT_EQ(15.0, z[0]);
  EXPECT_FLOAT_EQ(15.0, z[1]);
  EXPECT_EQ(2U,z.size());

  Matrix<double,Dynamic,Dynamic> m(2,3);
  fill(m,double(12));
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 3; ++j)
      EXPECT_FLOAT_EQ(12.0, m(i,j));
  
  Matrix<double,Dynamic,1> rv(3);
  fill(rv,double(13));
  for (int i = 0; i < 3; ++i)
    EXPECT_FLOAT_EQ(13.0, rv(i));

  Matrix<double,1,Dynamic> v(4);
  fill(v,double(22));
  for (int i = 0; i < 4; ++i)
    EXPECT_FLOAT_EQ(22.0, v(i));

  vector<vector<double> > d(3,vector<double>(2));
  fill(d,double(54));
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 2; ++j)
      EXPECT_FLOAT_EQ(54, d[i][j]);
}
TEST(AgradRevMatrix, fillDouble) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::fill;
  Matrix<double,Dynamic,1> y(3);
  fill(y,3.0);
  EXPECT_EQ(3,y.size());
  EXPECT_FLOAT_EQ(3.0,y[0]);
}
TEST(MathMatrix, fill_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(3,2);
  Eigen::MatrixXd m2(3,2);
  Eigen::VectorXd v1(3);
  Eigen::RowVectorXd v2(3);
  std::vector<double> v3(14, 1.1);
        
  using stan::math::fill;
  using boost::math::isnan;
  
  fill(m1, nan);
  fill(m2, nan);
  fill(v1, nan);
  fill(v2, nan);
  fill(v3, nan);
    
  expect_matrix_is_nan(m1);
  expect_matrix_is_nan(m2);
  expect_matrix_is_nan(v1);
  expect_matrix_is_nan(v2);
  expect_matrix_is_nan(v3);
}
