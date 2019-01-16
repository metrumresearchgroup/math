#include <stan/math/rev/mat.hpp>  // FIX ME - includes should be more specific
#include <test/unit/math/torsten/expect_near_matrix_eq.hpp>
#include <test/unit/math/torsten/expect_matrix_eq.hpp>
#include <test/unit/math/torsten/pk_twocpt_test_fixture.hpp>
#include <test/unit/math/torsten/util_generalOdeModel.hpp>
#include <stan/math/torsten/mpi/init.hpp>
#include <stan/math/torsten/PKModelTwoCpt.hpp>
#include <stan/math/torsten/pk_onecpt_model.hpp>
#include <stan/math/torsten/pk_twocpt_model.hpp>
#include <gtest/gtest.h>
#include <stan/math/rev/mat.hpp>  // FIX ME - include should be more specific
#include <test/unit/math/torsten/util_PKModelTwoCpt.hpp>
#include <vector>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;

TEST_F(TorstenPKTwoCptTest, MultipleDoses) {
  std::vector<Matrix<double, Dynamic, Dynamic> > x;

  x = torsten::PKModelTwoCpt(time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m,
                             pMatrix_m, biovar_m, tlag_m);

  Matrix<double, Dynamic, Dynamic> amounts(10, 3);
  amounts << 1000.0, 0.0, 0.0,
    740.818221, 238.3713, 12.75775,
    548.811636, 379.8439, 43.55827,
    406.569660, 455.3096, 83.95657,
    301.194212, 486.6965, 128.32332,
    223.130160, 489.4507, 173.01118,
    165.298888, 474.3491, 215.75441,
    122.456428, 448.8192, 255.23842,
    90.717953, 417.9001, 290.79297,
    8.229747, 200.8720, 441.38985;

  for (int i = 0; i < np; ++i) {
    expect_matrix_eq(amounts, x[i]);
 }
}

TEST_F(TorstenPKTwoCptTest, MultipleDoses2) {
  std::vector<Matrix<double, Dynamic, Dynamic> > x;

  x = torsten::PKModelTwoCpt(time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m,
                             pMatrix_m, biovar_m, tlag_m);

  Matrix<double, Dynamic, Dynamic> amounts(10, 3);
  amounts << 1000.0, 0.0, 0.0,
    740.818221, 238.3713, 12.75775,
    548.811636, 379.8439, 43.55827,
    406.569660, 455.3096, 83.95657,
    301.194212, 486.6965, 128.32332,
    223.130160, 489.4507, 173.01118,
    165.298888, 474.3491, 215.75441,
    122.456428, 448.8192, 255.23842,
    90.717953, 417.9001, 290.79297,
    8.229747, 200.8720, 441.38985;

  for (int i = 0; i < np; ++i) {
    expect_matrix_eq(amounts, x[i]);
 }
}
