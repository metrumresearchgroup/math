#include <stan/math/rev/mat.hpp>  // FIX ME - includes should be more specific
#include <gtest/gtest.h>
#include <test/unit/math/prim/mat/fun/expect_near_matrix_eq.hpp>
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <test/unit/math/torsten/util_generalOdeModel.hpp>
// #include <stan/math/torsten/event_list.hpp>
#include <stan/math/torsten/pk_system.hpp>
// #include <stan/math/torsten/PKModelTwoCpt2.hpp>

// TEST(Torsten, pk_eventhistory) {
//   using refactor::PKEvent;
//   PKEvent<double, double, double, double, double, double, double> ev;
//   auto ev1 = ev;
//   std::stringstream buffer;
//   buffer << ev << std::endl;
// }

// struct A{
//   int& x;
//   A(int& y) : x(y) {}
// };
//   void test_a(A a) {}

// template<typename T>
// void test_pk_parameter_vector_coersion(refactor::PKParameterVector<T> v) {}
// TEST(Torsten, pk_parameter_vector) {
//   int x =3;
//   test_a(x);
//   std::vector<int> v1;
//   std::vector<std::vector<int> > v2;
//   // test_pk_parameter_vector_coersion(v1);
//   // test_pk_parameter_vector_coersion(v2);XS
// }

TEST(Torsten, pk_system) {
  using std::vector;
  using Eigen::Matrix;
  using Eigen::Dynamic;

  vector<vector<double> > pMatrix(1);
  pMatrix[0].resize(5);
  pMatrix[0][0] = 5;  // CL
  pMatrix[0][1] = 8;  // Q
  pMatrix[0][2] = 20;  // Vc
  pMatrix[0][3] = 70;  // Vp
  pMatrix[0][4] = 1.2;  // ka

  vector<vector<double> > biovar(1);
  biovar[0].resize(3);
  biovar[0][0] = 1;  // F1
  biovar[0][1] = 1;  // F2
  biovar[0][2] = 1;  // F3

  vector<vector<double> > tlag(1);
  tlag[0].resize(3);
  tlag[0][0] = 0;  // tlag1
  tlag[0][1] = 0;  // tlag2
  tlag[0][2] = 0;  // tlag3

  vector<double> time(10);
  time[0] = 0.0;
  for(int i = 1; i < 9; i++) time[i] = time[i - 1] + 0.25;
  time[9] = 4.0;

  vector<double> amt(10, 0);
  amt[0] = 1000;

  vector<double> rate(10, 0);

  vector<int> cmt(10, 2);
  cmt[0] = 1;

  vector<int> evid(10, 0);
  evid[0] = 1;

  vector<double> ii(10, 0);
  ii[0] = 12;
	
  vector<int> addl(10, 0);
  addl[0] = 14;
	
  vector<int> ss(10, 0);

  double CL = 10, Vc = 80, ka = 1.2, k10 = CL / Vc;
  Matrix<double, Dynamic, Dynamic> system(2, 2);
  system << -ka, 0, ka, -k10;
  vector<Matrix<double, Dynamic, Dynamic> > system_array(1, system); 

  // Matrix<double, Dynamic, Dynamic> x;
  // x = torsten::PKModelTwoCpt2(time, amt, rate, ii, evid, cmt, addl, ss,
  //                             pMatrix, biovar, tlag);


  // refactor::PKEventList<double, double, double, double,
  //                       double, double, double> el{time, amt,
  //     rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag};

  // refactor::PKEventList<double, double, double, double,
  //                       double, double, double> el2{time, amt,
  //     rate, ii, evid, cmt, addl, ss, pMatrix[0], biovar[0], tlag};

  // refactor::PKEventList<double, double, double, double,
  //                       double, double, double> el3{time, amt,
  //     rate, ii, evid, cmt, addl, ss, pMatrix, biovar[0], tlag[0]};

  using model_type = refactor::PKOneCptModel<double, double, double, double>;

  refactor::PKSystem<double, double, double, double, double, double, double>
    pk1 (time, amt,
         rate, ii,
         evid, cmt,
         addl, ss,
         model_type::ncmt,
         pMatrix,
         biovar, tlag, system_array);
                                      
  refactor::PKSystem<double, double, double, double, double, double, double>
    pk2 (time, amt,
         rate, ii,
         evid, cmt,
         addl, ss,
         model_type::ncmt,
         pMatrix[0],
         biovar[0], tlag[0], system_array);

  {
  refactor::PKOneCptModelSolver sol;
  refactor::PKOneCptModelSolverSS ssol;
  auto res = pk1.solve_with<refactor::PKOneCptModelSolver,
                             refactor::PKOneCptModelSolverSS,
                             refactor::PKOneCptModel,
                             double, double, double, double
                             >(sol, ssol);
  }

  {
    refactor::PKTwoCptModelSolver sol;
    refactor::PKTwoCptModelSolverSS ssol;
    auto res = pk1.solve_with<refactor::PKTwoCptModelSolver,
                               refactor::PKTwoCptModelSolverSS,
                               refactor::PKTwoCptModel,
                               double, double, double, double
                               >(sol, ssol);
  }

  // pk1.solve_with<double, double, double, double, refactor::PKOneCptModel, 
  //                refactor::PKOneCptModelSolver, refactor::PKOneCptModelSolverSS>(sol, ssol);

  // Matrix<double, Dynamic, Dynamic> amounts(10, 3);
  // amounts << 1000.0, 0.0, 0.0,
  //   740.818221, 238.3713, 12.75775,
  //   548.811636, 379.8439, 43.55827,
  //   406.569660, 455.3096, 83.95657,
  //   301.194212, 486.6965, 128.32332,
  //   223.130160, 489.4507, 173.01118,
  //   165.298888, 474.3491, 215.75441,
  //   122.456428, 448.8192, 255.23842,
  //   90.717953, 417.9001, 290.79297,
  //   8.229747, 200.8720, 441.38985;

  // expect_matrix_eq(amounts, x);

}
