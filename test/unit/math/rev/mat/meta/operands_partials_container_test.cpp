#include <stan/math/prim/mat/meta/container_view.hpp>
#include <stan/math/rev/mat/meta/operands_partials_container.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/core.hpp>

TEST(AgradPartialsVari, OperandsPartialsContainer) {
  using stan::math::operands_partials_container;
  using stan::math::var;
  using Eigen::Dynamic;
  using Eigen::Matrix;
  Matrix<var, Dynamic, 1> m(10);
  Matrix<double, Dynamic, 1> d_m(10);
  Matrix<var, Dynamic, Dynamic> l(10, 10);
  Matrix<double, Dynamic, Dynamic> d_l(10, 10);

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

  operands_partials_container<Matrix<var, Dynamic, 1>, Matrix<double, Dynamic, 1>,
                              Matrix<var, Dynamic, Dynamic>, Matrix<double, Dynamic, Dynamic> > 
                                test(m, l);

  test.d_x1[0] = d_m;
  test.d_x2[0] = d_l;
}
//TEST(AgradPartialsVari,OperandsAndPartials1) {
//  using stan::math::vari;
//  using stan::math::OperandsAndPartials;
//  using stan::math::var;
//
//  var x = 2.0;
//  var z = -3.0 * x;  // dz/dx = -3
//  OperandsAndPartials<var> o(z);
//  o.d_x1[0] += 5.0;  // dy/dz = 5
//
//  var y = o.to_var(-1.0);
//
//  stan::math::grad(y.vi_);
//  EXPECT_FLOAT_EQ(-15.0, x.adj());  // dy/dx = -15
//}
//TEST(AgradPartialsVari,OperandsAndPartials2) {
//  using stan::math::OperandsAndPartials;
//  using stan::math::vari;
//  using stan::math::var;
//
//  var x1 = 2.0;
//  var x2 = 3.0;
//  var z1 = -5.0 * x1; // dz1/dx1 = -5
//  var z2 = -7.0 * x2; // dz2/dx2 = -7
//  OperandsAndPartials<var,var> o(z1,z2);
//  o.d_x1[0] += 11.0;  // dy/dz1 = 11.0
//  o.d_x2[0] += 13.0;  // dy/dz2 = 13.0
//  var y = o.to_var(-1.0);
//
//  stan::math::grad(y.vi_);
//  EXPECT_FLOAT_EQ(-55.0, x1.adj());  // dy/dx1 = -55
//  EXPECT_FLOAT_EQ(-91.0, x2.adj());  // dy/dx2 = -91
//}
//TEST(AgradPartialsVari, OperandsAndPartials3) {
//  using stan::math::OperandsAndPartials;
//  using stan::math::vari;
//  using stan::math::var;
//
//  var x1 = 2.0;
//  var x2 = 3.0;
//  var x3 = 5.0;
//  var z1 = -7.0 * x1;  // dz1/dx1 = -5
//  var z2 = -9.0 * x2;  // dz2/dx2 = -7
//  var z3 = -11.0 * x3; // dz3/dx3 = -11
//
//  OperandsAndPartials<var,var,var> o(z1, z2, z3);
//  o.d_x1[0] += 17.0;  // dy/dz1 = 17.0
//  o.d_x2[0] += 19.0;  // dy/dz2 = 19.0
//  o.d_x3[0] += 23.0;  // dy/dz3 = 23.0
//  var y = o.to_var(-1.0);
//
//  stan::math::grad(y.vi_);
//  EXPECT_FLOAT_EQ(-119.0, x1.adj());  // dy/dx1 = -119
//  EXPECT_FLOAT_EQ(-171.0, x2.adj());  // dy/dx2 = -133
//  EXPECT_FLOAT_EQ(-253.0, x3.adj());  // dy/dx2 = -253
//}
//TEST(AgradPartialsVari, OperandsAndPartials_check_throw) {
//  using stan::math::OperandsAndPartials;
//  using stan::math::var;
//  using std::vector;
//  
//  double d;
//  vector<double> D;
//  var v;
//  vector<var> V;
//  
//  OperandsAndPartials<> o1(d,d,d,d,d,d);
//  EXPECT_THROW(o1.d_x1[0], std::out_of_range);
//  EXPECT_THROW(o1.d_x2[0], std::out_of_range);
//  EXPECT_THROW(o1.d_x3[0], std::out_of_range);
//  EXPECT_THROW(o1.d_x4[0], std::out_of_range);
//  EXPECT_THROW(o1.d_x5[0], std::out_of_range);
//  EXPECT_THROW(o1.d_x6[0], std::out_of_range);
//
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
//}
