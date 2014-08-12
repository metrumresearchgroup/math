#include <stan/math/matrix/resize.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

using Eigen::Matrix;
using Eigen::Dynamic;

using stan::math::matrix_d;
using stan::math::vector_d;
using stan::math::row_vector_d;

TEST(MathMatrix, resize_double) {
  double x = 5;
  std::vector<size_t> dims;
  stan::math::resize(x,dims);
}
TEST(MathMatrix, resize_svec_double) {
  std::vector<double> y;
  std::vector<size_t> dims;
  EXPECT_EQ(0U, y.size());

  dims.push_back(4U);
  stan::math::resize(y,dims);
  EXPECT_EQ(4U, y.size());

  dims[0] = 2U;
  stan::math::resize(y,dims);
  EXPECT_EQ(2U, y.size());
}
TEST(MathMatrix, resize_vec_double) {
  Matrix<double,Dynamic,1> v(2);
  std::vector<size_t> dims;
  EXPECT_EQ(2, v.size());

  dims.push_back(17U);
  stan::math::resize(v,dims);
  EXPECT_EQ(17, v.size());

  dims[0] = 3U;
  stan::math::resize(v,dims);
  EXPECT_EQ(3, v.size());
}
TEST(MathMatrix, resize_rvec_double) {
  Matrix<double,1,Dynamic> rv(2);
  std::vector<size_t> dims;
  EXPECT_EQ(2, rv.size());

  dims.push_back(17U);
  stan::math::resize(rv,dims);
  EXPECT_EQ(17, rv.size());

  dims[0] = 3U;
  stan::math::resize(rv,dims);
  EXPECT_EQ(3, rv.size());
}
TEST(MathMatrix, resize_mat_double) {
  Matrix<double,Dynamic,Dynamic> m(2,3);
  std::vector<size_t> dims;
  EXPECT_EQ(2, m.rows());
  EXPECT_EQ(3, m.cols());

  dims.push_back(7U);
  dims.push_back(17U);
  stan::math::resize(m,dims);
  EXPECT_EQ(7, m.rows());
  EXPECT_EQ(17, m.cols());
}
TEST(MathMatrix, resize_svec_svec_double) {
  std::vector<std::vector<double> > xx;
  EXPECT_EQ(0U,xx.size());
  std::vector<size_t> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  stan::math::resize(xx,dims);
  EXPECT_EQ(4U,xx.size());
  EXPECT_EQ(5U,xx[0].size());

  dims[0] = 3U;
  dims[1] = 7U;
  stan::math::resize(xx,dims);
  EXPECT_EQ(3U,xx.size());
  EXPECT_EQ(7U,xx[1].size());  
}
TEST(MathMatrix, resize_svec_v_double) {
  std::vector<Matrix<double,Dynamic,1> > xx;
  EXPECT_EQ(0U,xx.size());
  std::vector<size_t> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  stan::math::resize(xx,dims);
  EXPECT_EQ(4U,xx.size());
  EXPECT_EQ(5,xx[0].size());

  dims[0] = 3U;
  dims[1] = 7U;
  stan::math::resize(xx,dims);
  EXPECT_EQ(3U,xx.size());
  EXPECT_EQ(7,xx[1].size());  
}
TEST(MathMatrix, resize_svec_rv_double) {
  std::vector<Matrix<double,1,Dynamic> > xx;
  EXPECT_EQ(0U,xx.size());
  std::vector<size_t> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  stan::math::resize(xx,dims);
  EXPECT_EQ(4U,xx.size());
  EXPECT_EQ(5,xx[0].size());

  dims[0] = 3U;
  dims[1] = 7U;
  stan::math::resize(xx,dims);
  EXPECT_EQ(3U,xx.size());
  EXPECT_EQ(7,xx[1].size());  
}
TEST(MathMatrix, resize_svec_svec_matrix_double) {
  std::vector<std::vector<Matrix<double,Dynamic,Dynamic> > > mm;
  std::vector<size_t> dims;
  dims.push_back(4U);
  dims.push_back(5U);
  dims.push_back(6U);
  dims.push_back(3U);
  stan::math::resize(mm,dims);
  EXPECT_EQ(4U,mm.size());
  EXPECT_EQ(5U,mm[0].size());
  EXPECT_EQ(6,mm[1][2].rows());
  EXPECT_EQ(3,mm[3][4].cols());
}


TEST(MathMatrix, resize_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::MatrixXd m1(4,3);
  m1 << 1, nan, 1.1,
        3, 4.1, 2.1,
        2, 0.1, 5.1,
        nan, 6, nan;

  std::vector<double> v2(14, 1.1);
  v2[0] = 1.2;
  v2[1] = nan;
  
  using stan::math::resize;
  using boost::math::isnan;
  
  std::vector<size_t> dim_(2, 2);
  dim_[1] = 6;
  resize(m1, dim_); //2 x 6
  EXPECT_EQ(1, m1(0, 0));
  EXPECT_EQ(3, m1(1, 0));
  EXPECT_EQ(2, m1(0, 1));
  EXPECT_PRED1(isnan<double>, m1(1, 1));
  EXPECT_PRED1(isnan<double>, m1(0, 2));
  EXPECT_EQ(4.1, m1(1, 2));
  EXPECT_EQ(0.1, m1(0, 3));
  EXPECT_EQ(6, m1(1, 3));
  EXPECT_EQ(1.1, m1(0, 4));
  EXPECT_EQ(2.1, m1(1, 4));
  EXPECT_EQ(5.1, m1(0, 5));
  EXPECT_PRED1(isnan<double>, m1(1, 5));
  
  resize(v2, std::vector<size_t>(1, 3));
  EXPECT_EQ(1.2, v2[0]);
  EXPECT_PRED1(isnan<double>, v2[1]);
  EXPECT_EQ(1.1, v2[2]);    
}
