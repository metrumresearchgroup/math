#include <stan/math/prim/mat/meta/container_view.hpp>
#include <stan/math/rev/mat/meta/operands_partials_container.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>

TEST(AgradPartialsVari, operands_partials_container_vec_mat) {
  using stan::math::operands_partials_container;
  using stan::math::var;
  using Eigen::Matrix;
  Matrix<var, -1, 1> m(10);
  Matrix<double, -1, 1> d_m(10);
  Matrix<var, -1, -1> l(10, 10);
  Matrix<double, -1, -1> d_l(10, 10);

  for (int i = 0; i < 10; ++i) {
    m(i) = i;
    d_m(i) = i + 1.0;
  }

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      l(i,j) = i + j;
      d_l(i,j) = i + j + 1.0;
    }
  }

  operands_partials_container<Matrix<var, -1, 1>, Matrix<double, -1, 1>,
                              Matrix<var, -1, -1>, Matrix<double, -1, -1> > 
                                test(m, l);

  test.d_x1[0] = d_m;
  test.d_x2[0] = d_l;
  var y = test.to_var(-1.0, m, l);
  stan::math::grad(y.vi_);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) 
      EXPECT_FLOAT_EQ(i + j + 1.0, l(i,j).adj());  // dy/dx = -15
    EXPECT_FLOAT_EQ(i + 1.0, m(i).adj());
  }
}
TEST(AgradPartialsVari, operands_partials_container_vec_mat_scal) {
  using stan::math::operands_partials_container;
  using stan::math::var;
  using Eigen::Matrix;
  Matrix<var, -1, 1> m(10);
  Matrix<double, -1, 1> d_m(10);
  Matrix<var, -1, -1> l(10, 10);
  Matrix<double, -1, -1> d_l(10, 10);
  var x = 1.0;

  for (int i = 0; i < 10; ++i) {
    m(i) = i;
    d_m(i) = i + 1.0;
  }

  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      l(i,j) = i + j;
      d_l(i,j) = i + j + 1.0;
    }
  }

  operands_partials_container<Matrix<var, -1, 1>, Matrix<double, -1, 1>,
                              Matrix<var, -1, -1>, Matrix<double, -1, -1>,
                              var, double> test(m, l, x);

  test.d_x1[0] = d_m;
  test.d_x2[0] = d_l;
  test.d_x3[0] += 5.0;
  var y = test.to_var(-1.0, m, l);
  stan::math::grad(y.vi_);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) 
      EXPECT_FLOAT_EQ(i + j + 1.0, l(i,j).adj());  // dy/dx = -15
    EXPECT_FLOAT_EQ(i + 1.0, m(i).adj());
  }
  EXPECT_FLOAT_EQ(5.0, x.adj());
}
TEST(AgradPartialsVari,operands_partials_container_scal) {
  using stan::math::operands_partials_container;
  using stan::math::var;

  var x = 2.0;
  var z = -3.0 * x;  // dz/dx = -3
  operands_partials_container<var,double> o(z);
  o.d_x1[0] += 5.0;  // dy/dz = 5

  var y = o.to_var(-1.0, z);

  stan::math::grad(y.vi_);
  EXPECT_FLOAT_EQ(-15.0, x.adj());  // dy/dx = -15
}
TEST(AgradPartialsVari,operands_partials_container_scal_2) {
  using stan::math::operands_partials_container;
  using stan::math::vari;
  using stan::math::var;

  var x1 = 2.0;
  var x2 = 3.0;
  var z1 = -5.0 * x1; // dz1/dx1 = -5
  var z2 = -7.0 * x2; // dz2/dx2 = -7
  operands_partials_container<var,double,
                              var,double> o(z1,z2);
  o.d_x1[0] += 11.0;  // dy/dz1 = 11.0
  o.d_x2[0] += 13.0;  // dy/dz2 = 13.0
  var y = o.to_var(-1.0, z1, z2);

  stan::math::grad(y.vi_);
  EXPECT_FLOAT_EQ(-55.0, x1.adj());  // dy/dx1 = -55
  EXPECT_FLOAT_EQ(-91.0, x2.adj());  // dy/dx2 = -91
}
TEST(AgradPartialsVari, operands_partials_container_scal_3) {
  using stan::math::operands_partials_container;
  using stan::math::var;

  var x1 = 2.0;
  var x2 = 3.0;
  var x3 = 5.0;
  var z1 = -7.0 * x1;  // dz1/dx1 = -7
  var z2 = -9.0 * x2;  // dz2/dx2 = -9
  var z3 = -11.0 * x3; // dz3/dx3 = -11

  operands_partials_container<var, double,
                              var, double,
                              var, double> o(z1, z2, z3);
  o.d_x1[0] += 17.0;  // dy/dz1 = 17.0
  o.d_x2[0] += 19.0;  // dy/dz2 = 19.0
  o.d_x3[0] += 23.0;  // dy/dz3 = 23.0
  var y = o.to_var(-1.0, z1, z2, z3);

  stan::math::grad(y.vi_);
  EXPECT_FLOAT_EQ(-119.0, x1.adj());  // dy/dx1 = -119
  EXPECT_FLOAT_EQ(-171.0, x2.adj());  // dy/dx2 = -133
  EXPECT_FLOAT_EQ(-253.0, x3.adj());  // dy/dx2 = -253
}
TEST(AgradPartialsVari, operands_partials_container_scal_4) {
  using stan::math::operands_partials_container;
  using stan::math::var;

  var x1 = 2.0;
  var x2 = 3.0;
  var x3 = 5.0;
  var z1 = -7.0 * x1;  // dz1/dx1 = -5
  var z2 = -9.0 * x2;  // dz2/dx2 = -7
  var z3 = -11.0 * x3; // dz3/dx3 = -11

  operands_partials_container<var, double,
                              var, double,
                              var, double> o(z1, z2, z3);
  o.d_x1[0] += 17.0;  // dy/dz1 = 17.0
  o.d_x2[0] += 19.0;  // dy/dz2 = 19.0
  o.d_x3[0] += 23.0;  // dy/dz3 = 23.0
  var y = o.to_var(-1.0, z1, z2, z3);

  stan::math::grad(y.vi_);
  EXPECT_FLOAT_EQ(-119.0, x1.adj());  // dy/dx1 = -119
  EXPECT_FLOAT_EQ(-171.0, x2.adj());  // dy/dx2 = -133
  EXPECT_FLOAT_EQ(-253.0, x3.adj());  // dy/dx2 = -253
}
TEST(AgradPartialsVari, operators_partials_container_check_throw) {
  using stan::math::operands_partials_container;
  using stan::math::var;
  using std::vector;
  
  double d;
  vector<double> D;
  var v;
  vector<var> V;
  
  operands_partials_container<double, double,
                              double, double,
                              double, double,
                              double, double,
                              double, double,
                              double, double> o1(d,d,d,d,d,d);
  EXPECT_THROW(o1.d_x1[0], std::out_of_range);
  EXPECT_THROW(o1.d_x2[0], std::out_of_range);
  EXPECT_THROW(o1.d_x3[0], std::out_of_range);
  EXPECT_THROW(o1.d_x4[0], std::out_of_range);
  EXPECT_THROW(o1.d_x5[0], std::out_of_range);
  EXPECT_THROW(o1.d_x6[0], std::out_of_range);

//  OperandsAndPartials<var,var,var,var,var,var> o2(v,v,v,v,v,v);
//  EXPECT_NO_THROW(o2.d_x1[0]);
//  EXPECT_NO_THROW(o2.d_x2[0]);
//  EXPECT_NO_THROW(o2.d_x3[0]);
//  EXPECT_NO_THROW(o2.d_x4[0]);
//  EXPECT_NO_THROW(o2.d_x5[0]);
//  EXPECT_NO_THROW(o2.d_x6[0]);
//
//  OperandsAndPartials<vector<double>,vector<double>,vector<double>,
//                      vector<double>,vector<double>,vector<double> > o3(D,D,D,D,D,D);
//  EXPECT_THROW(o3.d_x1[0], std::out_of_range);
//  EXPECT_THROW(o3.d_x2[0], std::out_of_range);
//  EXPECT_THROW(o3.d_x3[0], std::out_of_range);
//  EXPECT_THROW(o3.d_x4[0], std::out_of_range);
//  EXPECT_THROW(o3.d_x5[0], std::out_of_range);
//  EXPECT_THROW(o3.d_x6[0], std::out_of_range);
//
//  OperandsAndPartials<vector<var>,vector<var>,vector<var>,
//                      vector<var>,vector<var>,vector<var> > o4(V,V,V,V,V,V);
//  EXPECT_NO_THROW(o4.d_x1[0]);
//  EXPECT_NO_THROW(o4.d_x2[0]);
//  EXPECT_NO_THROW(o4.d_x3[0]);
//  EXPECT_NO_THROW(o4.d_x4[0]);
//  EXPECT_NO_THROW(o4.d_x5[0]);
//  EXPECT_NO_THROW(o4.d_x6[0]);
}
