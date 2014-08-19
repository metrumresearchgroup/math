#include <stan/agrad/rev/matrix.hpp>
#include <stan/math/matrix/cholesky_decompose.hpp>
#include <gtest/gtest.h>
#include <test/unit/agrad/util.hpp>
#include <stan/math/matrix/singular_values.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <stan/agrad/rev/functions/sqrt.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>

TEST(AgradRevMatrix,mat_cholesky) {
  using stan::agrad::matrix_v;
  using stan::math::transpose;
  using stan::math::cholesky_decompose;
  using stan::math::singular_values;

  // symmetric
  matrix_v X(2,2);
  AVAR a = 3.0;
  AVAR b = -1.0;
  AVAR c = -1.0;
  AVAR d = 1.0;
  X << a, b, 
    c, d;
  
  matrix_v L = cholesky_decompose(X);

  matrix_v LL_trans = multiply(L,transpose(L));
  EXPECT_FLOAT_EQ(a.val(),LL_trans(0,0).val());
  EXPECT_FLOAT_EQ(b.val(),LL_trans(0,1).val());
  EXPECT_FLOAT_EQ(c.val(),LL_trans(1,0).val());
  EXPECT_FLOAT_EQ(d.val(),LL_trans(1,1).val());

  EXPECT_NO_THROW(singular_values(X));
}
TEST(AgradRevMatrix,mat_cholesky_nan) {
  using stan::agrad::matrix_v;
  using stan::math::transpose;
  using stan::math::cholesky_decompose;
  using stan::math::singular_values;
  double nan = std::numeric_limits<double>::quiet_NaN();

  // symmetric
  matrix_v X(2,2);
  AVAR a = 3.0;
  AVAR b = nan;
  AVAR c = -1.0;
  AVAR d = 1.0;
  X << a, b, 
    c, d;
  
  matrix_v L = cholesky_decompose(X);

  matrix_v LL_trans = multiply(L,transpose(L));
  EXPECT_FLOAT_EQ(a.val(),LL_trans(0,0).val());
  EXPECT_FLOAT_EQ(c.val(),LL_trans(0,1).val());
  EXPECT_FLOAT_EQ(c.val(),LL_trans(1,0).val());
  EXPECT_FLOAT_EQ(d.val(),LL_trans(1,1).val());

  EXPECT_NO_THROW(singular_values(X));

  X << a,c,b,d;
  L = cholesky_decompose(X);
  LL_trans = multiply(L,transpose(L));
  EXPECT_FLOAT_EQ(a.val(),LL_trans(0,0).val());
  EXPECT_TRUE(boost::math::isnan(LL_trans(0,1).val()));
  EXPECT_TRUE(boost::math::isnan(LL_trans(1,0).val()));
  EXPECT_TRUE(boost::math::isnan(LL_trans(1,1).val()));
}

