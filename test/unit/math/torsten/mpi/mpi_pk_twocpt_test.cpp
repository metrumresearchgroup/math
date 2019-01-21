#include <stan/math/rev/mat.hpp>  // FIX ME - includes should be more specific
#include <test/unit/math/torsten/expect_near_matrix_eq.hpp>
#include <test/unit/math/torsten/expect_matrix_eq.hpp>
#include <test/unit/math/torsten/pk_twocpt_test_fixture.hpp>
#include <test/unit/math/torsten/util_generalOdeModel.hpp>
#include <test/unit/math/torsten/test_util.hpp>
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
using stan::math::var;

// TEST_F(TorstenPKTwoCptTest, MultipleDoses) {
//   std::vector<Matrix<double, Dynamic, Dynamic> > x;

//   x = torsten::PKModelTwoCpt(time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m,
//                              pMatrix_m, biovar_m, tlag_m);

//   Matrix<double, Dynamic, Dynamic> amounts(10, 3);
//   amounts << 1000.0, 0.0, 0.0,
//     740.818221, 238.3713, 12.75775,
//     548.811636, 379.8439, 43.55827,
//     406.569660, 455.3096, 83.95657,
//     301.194212, 486.6965, 128.32332,
//     223.130160, 489.4507, 173.01118,
//     165.298888, 474.3491, 215.75441,
//     122.456428, 448.8192, 255.23842,
//     90.717953, 417.9001, 290.79297,
//     8.229747, 200.8720, 441.38985;

//   for (int i = 0; i < np; ++i) {
//     expect_matrix_eq(amounts, x[i]);
//  }
// }

// TEST_F(TorstenPKTwoCptTest, multiple_bolus_doses_population) {
//   {
//     vector<vector<vector<var> > > pMatrix_m_v(pMatrix_m.size());
//     vector<vector<var> > pMatrix_v = torsten::test::to_var(pMatrix);
//     for (int i = 0; i < np; ++i) {
//       pMatrix_m_v[i] = torsten::test::to_var(pMatrix_m[i]);
//     }
//     // vector<Matrix<var, Dynamic, Dynamic> > x_m(torsten::PKModelTwoCpt(time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m_v, biovar_m, tlag_m)); // NOLINT
//     Matrix<var, Dynamic, Dynamic> x_m;
//     torsten::PKModelTwoCpt(x_m, time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m_v, biovar_m, tlag_m); // NOLINT
//     std::cout << "taki test: " << x_m.size() << "\n";
//     Matrix<var, Dynamic, Dynamic> x = torsten::PKModelTwoCpt(time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT
//     for (size_t i = 0; i < np; ++i) {
//       for (int j = 0; j < x_m.size(); ++j) {
//         std::cout << "taki test: " << x_m(j).val() << "\n";
//       }
//       // torsten::test::test_grad(pMatrix_m_v[i][0], pMatrix_v[0], x_m[i], x, 1.E-8, 1.E-5);
//     }
//   }
// }

TEST_F(TorstenPKTwoCptTest, multiple_bolus_doses_population) {
  {
    vector<vector<vector<var> > > pMatrix_m_v(pMatrix_m.size());
    vector<vector<var> > pMatrix_v(torsten::test::to_var(pMatrix));
    for (int i = 0; i < np; ++i) {
      pMatrix_m_v[i] = torsten::test::to_var(pMatrix_m[i]);
    }

    Matrix<var, Dynamic, Dynamic> x =
      torsten::PKModelTwoCpt(time, amt, rate, ii, evid, cmt, addl, ss, pMatrix_v, biovar, tlag); // NOLINT
    vector<Matrix<var, Dynamic, Dynamic> > x_m =
      torsten::PKModelTwoCpt(time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m_v, biovar_m, tlag_m); // NOLINT

    for (int i = 0; i < np; ++i) {
      torsten::test::test_grad(pMatrix_m_v[i][0], pMatrix_v[0], x_m[i], x, 1.E-8, 1.E-5);
    }
  }
}

TEST_F(TorstenPKTwoCptTest, multiple_bolus_doses_population_perf) {
    vector<vector<vector<var> > > pMatrix_m_v(pMatrix_m.size());
    vector<vector<var> > pMatrix_v(torsten::test::to_var(pMatrix));
    for (int i = 0; i < np; ++i) {
      pMatrix_m_v[i] = torsten::test::to_var(pMatrix_m[i]);
    }

    vector<Matrix<var, Dynamic, Dynamic> > x_m =
      torsten::PKModelTwoCpt(time_m, amt_m, rate_m, ii_m, evid_m, cmt_m, addl_m, ss_m, pMatrix_m_v, biovar_m, tlag_m); // NOLINT
}
