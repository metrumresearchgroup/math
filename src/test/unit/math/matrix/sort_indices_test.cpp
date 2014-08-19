#include <stdexcept>
#include <stan/math/matrix/sort_indices.hpp>
#include <gtest/gtest.h>
#include <boost/math/special_functions/fpclassify.hpp>

template <typename T>
void test_sort_indices_asc() {
  using stan::math::sort_indices_asc;

  T c(1);
  c[0] = 1.7;
  std::vector<int> d = sort_indices_asc(c);
  EXPECT_EQ(c.size(), typename T::size_type(d.size()));
  EXPECT_EQ(d.at(0), 1);

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  std::vector<int> f = sort_indices_asc(e);
  EXPECT_EQ(e.size(), typename T::size_type(f.size()));
  EXPECT_EQ(f.at(0), 2);
  EXPECT_EQ(f.at(1), 1);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  std::vector<int> h = sort_indices_asc(g);
  EXPECT_EQ(g.size(), typename T::size_type(h.size()));
  EXPECT_EQ(h.at(0), 2);
  EXPECT_EQ(h.at(1), 1);
  EXPECT_EQ(h.at(2), 3);

  T z; 
  EXPECT_NO_THROW(sort_indices_asc(z));
  EXPECT_EQ(typename T::size_type(0), z.size());
}

TEST(MathMatrix,sort_indices_asc) {
  using stan::math::sort_indices_asc;

  EXPECT_EQ(0U, sort_indices_asc(std::vector<int>(0)).size());

  test_sort_indices_asc<std::vector<double> >();
  test_sort_indices_asc<Eigen::Matrix<double,Eigen::Dynamic,1> >();
  test_sort_indices_asc<Eigen::Matrix<double,1,Eigen::Dynamic> >();
}


template <typename T>
void test_sort_indices_desc() {
  using stan::math::sort_indices_desc;

  T c(1);
  c[0] = 1.7;
  std::vector<int> d = sort_indices_desc(c);
  EXPECT_EQ(c.size(), typename T::size_type(d.size()));
  EXPECT_EQ(d.at(0), 1);

  T e(2);
  e[0] = 5.9;  e[1] = -1.2;
  std::vector<int> f = sort_indices_desc(e);
  EXPECT_EQ(e.size(), typename T::size_type(f.size()));
  EXPECT_EQ(f.at(0), 1);
  EXPECT_EQ(f.at(1), 2);

  T g(3);
  g[0] = 5.9;  g[1] = -1.2;   g[2] = 192.13456;
  std::vector<int> h = sort_indices_desc(g);
  EXPECT_EQ(g.size(), typename T::size_type(h.size()));
  EXPECT_EQ(h.at(0), 3);
  EXPECT_EQ(h.at(1), 1);
  EXPECT_EQ(h.at(2), 2);

  T z; 
  EXPECT_NO_THROW(sort_indices_desc(z));
  EXPECT_EQ(typename T::size_type(0), z.size());

}

TEST(MathMatrix,sort_indices_desc) {
  using stan::math::sort_indices_desc;

  EXPECT_EQ(0U, sort_indices_desc(std::vector<int>(0)).size());

  test_sort_indices_desc<std::vector<double> >();
  test_sort_indices_desc<Eigen::Matrix<double,Eigen::Dynamic,1> >();
  test_sort_indices_desc<Eigen::Matrix<double,1,Eigen::Dynamic> >();
}

TEST(MathMatrix, sort_indices_nan) {
  double nan = std::numeric_limits<double>::quiet_NaN();
  using stan::math::sort_indices_asc;
  using stan::math::sort_indices_desc;

  std::vector<double> a;
  a.push_back(nan); a.push_back(2); a.push_back(2); a.push_back(3);
  std::vector<int> res_asc = sort_indices_asc(a);
  std::vector<int> res_desc = sort_indices_desc(a);
  EXPECT_FLOAT_EQ(1, res_asc[0]);
  EXPECT_FLOAT_EQ(2, res_asc[1]);
  EXPECT_FLOAT_EQ(3, res_asc[2]);
  EXPECT_FLOAT_EQ(4, res_asc[3]);
  EXPECT_FLOAT_EQ(1, res_desc[0]);
  EXPECT_FLOAT_EQ(4, res_desc[1]);
  EXPECT_FLOAT_EQ(2, res_desc[2]);
  EXPECT_FLOAT_EQ(3, res_desc[3]);

  Eigen::RowVectorXd vec1(4);
  vec1 << nan, 2.0, 2.0, 3.0;
  std::vector<int> res_asc1 = sort_indices_asc(vec1);
  std::vector<int> res_desc1 = sort_indices_desc(vec1);
  EXPECT_FLOAT_EQ(1, res_asc1[0]);
  EXPECT_FLOAT_EQ(2, res_asc1[1]);
  EXPECT_FLOAT_EQ(3, res_asc1[2]);
  EXPECT_FLOAT_EQ(4, res_asc1[3]);
  EXPECT_FLOAT_EQ(1, res_desc1[0]);
  EXPECT_FLOAT_EQ(4, res_desc1[1]);
  EXPECT_FLOAT_EQ(2, res_desc1[2]);
  EXPECT_FLOAT_EQ(3, res_desc1[3]);

  Eigen::VectorXd vec3(4);
  vec3 << nan, 2.0, 2.0, 3.0;
  std::vector<int> res_asc2 = sort_indices_asc(vec3);
  std::vector<int> res_desc2 = sort_indices_desc(vec3);
  EXPECT_FLOAT_EQ(1, res_asc2[0]);
  EXPECT_FLOAT_EQ(2, res_asc2[1]);
  EXPECT_FLOAT_EQ(3, res_asc2[2]);
  EXPECT_FLOAT_EQ(4, res_asc2[3]);
  EXPECT_FLOAT_EQ(1, res_desc2[0]);
  EXPECT_FLOAT_EQ(4, res_desc2[1]);
  EXPECT_FLOAT_EQ(2, res_desc2[2]);
  EXPECT_FLOAT_EQ(3, res_desc2[3]);
}
