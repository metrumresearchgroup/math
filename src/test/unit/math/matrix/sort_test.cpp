#include <stdexcept>
#include <stan/math/matrix/sort.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

template <typename T>
void test_sort_asc() {
  using stan::math::sort_asc;

  T c(1);
  c[0] = 1.7;
  T d = sort_asc(c);
  EXPECT_EQ(c.size(), d.size());
  EXPECT_EQ(c[0], d[0]);

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  T f = sort_asc(e);
  EXPECT_EQ(e.size(), f.size());
  EXPECT_EQ(e[0], f[1]);
  EXPECT_EQ(e[1], f[0]);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  T h = sort_asc(g);
  EXPECT_EQ(g.size(), h.size());
  EXPECT_EQ(g[0], h[1]);
  EXPECT_EQ(g[1], h[0]);
  EXPECT_EQ(g[2], h[2]);

  T z; 
  EXPECT_NO_THROW(sort_asc(z));
  EXPECT_EQ(typename T::size_type(0), z.size());

}

TEST(MathMatrix,sort_asc) {
  using stan::math::sort_asc;

  EXPECT_EQ(0U, sort_asc(std::vector<int>(0)).size());

  test_sort_asc<std::vector<double> >();
  test_sort_asc<Eigen::Matrix<double,Eigen::Dynamic,1> >();
  test_sort_asc<Eigen::Matrix<double,1,Eigen::Dynamic> >();
}



template <typename T>
void test_sort_desc() {
  using stan::math::sort_desc;

  T c(1);
  c[0] = -1.7;
  T d = sort_desc(c);
  EXPECT_EQ(c.size(), d.size());
  EXPECT_EQ(c[0], d[0]);

  T e(2);
  e[0] = -5.9;  e[1] = 1.2;
  T f = sort_desc(e);
  EXPECT_EQ(e.size(), f.size());
  EXPECT_EQ(e[0], f[1]);
  EXPECT_EQ(e[1], f[0]);

  T g(3);
  g[0] = -5.9;  g[1] = 1.2;   g[2] = -192.13456;
  T h = sort_desc(g);
  EXPECT_EQ(g.size(), h.size());
  EXPECT_EQ(g[0], h[1]);
  EXPECT_EQ(g[1], h[0]);
  EXPECT_EQ(g[2], h[2]);
  
  T z; 
  EXPECT_NO_THROW(sort_desc(z));
  EXPECT_EQ(typename T::size_type(0), z.size());
}

TEST(MathMatrix,sort_desc) {
  using stan::math::sort_desc;    

  EXPECT_EQ(0U, sort_desc(std::vector<int>(0)).size());
  
  test_sort_desc<std::vector<double> >();
  test_sort_desc<Eigen::Matrix<double,Eigen::Dynamic,1> >();
  test_sort_desc<Eigen::Matrix<double,1,Eigen::Dynamic> >();
}

TEST(MathMatrix, sort_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  using stan::math::sort_asc;
  using stan::math::sort_desc;

  std::vector<double> a;
  a.push_back(nan); a.push_back(2); a.push_back(2); a.push_back(3);
  std::vector<double> res_asc = sort_asc(a);
  std::vector<double> res_desc = sort_desc(a);
  EXPECT_TRUE(boost::math::isnan(res_asc[0]));
  EXPECT_TRUE(boost::math::isnan(res_desc[0]));
  EXPECT_FLOAT_EQ(2, res_asc[1]);
  EXPECT_FLOAT_EQ(2, res_asc[2]);
  EXPECT_FLOAT_EQ(3, res_asc[3]);
  EXPECT_FLOAT_EQ(3, res_desc[1]);
  EXPECT_FLOAT_EQ(2, res_desc[2]);
  EXPECT_FLOAT_EQ(2, res_desc[3]);

  Eigen::RowVectorXd vec1(4);
  vec1 << nan, -33.1, 2.1, -33.1;
  Eigen::RowVectorXd res_asc1 = sort_asc(vec1);
  Eigen::RowVectorXd res_desc1 = sort_desc(vec1);
  EXPECT_TRUE(boost::math::isnan(res_asc1(0)));
  EXPECT_TRUE(boost::math::isnan(res_desc1(0)));
  EXPECT_FLOAT_EQ(-33.1, res_asc1(1));
  EXPECT_FLOAT_EQ(-33.1, res_asc1(2));
  EXPECT_FLOAT_EQ(2.1, res_asc1(3));
  EXPECT_FLOAT_EQ(2.1, res_desc1(1));
  EXPECT_FLOAT_EQ(-33.1, res_desc1(2));
  EXPECT_FLOAT_EQ(-33.1, res_desc1(3));

  Eigen::VectorXd vec3(4);
  vec3 << nan, -33.1, 2.1, -33.1;
  Eigen::VectorXd res_asc2 = sort_asc(vec3);
  Eigen::VectorXd res_desc2 = sort_desc(vec3);
  EXPECT_TRUE(boost::math::isnan(res_asc2(0)));
  EXPECT_TRUE(boost::math::isnan(res_desc2(0)));
  EXPECT_FLOAT_EQ(-33.1, res_asc2(1));
  EXPECT_FLOAT_EQ(-33.1, res_asc2(2));
  EXPECT_FLOAT_EQ(2.1, res_asc2(3));
  EXPECT_FLOAT_EQ(2.1, res_desc2(1));
  EXPECT_FLOAT_EQ(-33.1, res_desc2(2));
  EXPECT_FLOAT_EQ(-33.1, res_desc2(3));
}
