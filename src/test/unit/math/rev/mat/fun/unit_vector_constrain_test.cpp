#include <stan/math/rev/scal/fun/cos.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/sin.hpp>
#include <stan/math/prim/mat/fun/unit_vector_constrain.hpp>
#include <test/unit/math/rev/mat/fun/jacobian.hpp>

#include <gtest/gtest.h>

using Eigen::Matrix;
using Eigen::Dynamic;

TEST(MathRev, unit_vector_constrain_jacobian) {
  using stan::agrad::var;
  using std::vector;
  var a = 2.0;
  var b = 3.0;
  var c = -1.0;
  
  Matrix<var,Dynamic,1> y(3);
  y << a, b, c;
  
  var lp(0);
  Matrix<var,Dynamic,1> x 
    = stan::prob::unit_vector_constrain(y,lp);
  
  vector<var> indeps;
  indeps.push_back(a);
  indeps.push_back(b);
  indeps.push_back(c);

  vector<var> deps;
  deps.push_back(x(0));
  deps.push_back(x(1));
  deps.push_back(x(2));
  deps.push_back(x(3));
  
  vector<vector<double> > jacobian;
  stan::agrad::jacobian(deps,indeps,jacobian);

  Matrix<double,Dynamic,Dynamic> J(4,4);
  for (int m = 0; m < 4; ++m) {
    for (int n = 0; n < 3; ++n) {
      J(m,n) = jacobian[m][n];
    }
    J(m,3) = x(m).val(); 
  }
  
  double det_J = J.determinant();
  double log_det_J = log(fabs(det_J));

  EXPECT_FLOAT_EQ(log_det_J, lp.val()) << "J = " << J << std::endl << "det_J = " << det_J;
}
