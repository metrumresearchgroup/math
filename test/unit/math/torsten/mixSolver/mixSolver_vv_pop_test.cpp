#include <stan/math/rev/mat.hpp>
#include <stan/math/torsten/PKModel/Pred/fTwoCpt.hpp>
#include <stan/math/torsten/PKModel/Pred/fOneCpt.hpp>
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <test/unit/math/prim/mat/fun/expect_near_matrix_eq.hpp>
#include <gtest/gtest.h>
#include <ostream>
#include <fstream>
#include <vector>


struct FK_functor {
  /**
   * parms contains both the PK and the PD parameters.
   * x contains both the PK and the PD states.
   */
  template <typename T0, typename T1, typename T2, typename T3>
  inline
  std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
  operator()(const T0& t,
             const std::vector<T1>& x,
             const std::vector<T2>& parms,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream__) const {

    using Eigen::Matrix;
    using Eigen::Dynamic;
    typedef typename boost::math::tools::promote_args<T0, T1, T2, T3>::type scalar;
    int nParms = 9;
    // PK variables
    scalar CL, Q, VC, VP, ka, k10, k12, k21;

    // PD variables
    scalar MTT, circ0, alpha, gamma, ktr, prol, transit1, transit2, transit3,
      circ, conc, Edrug;

    int nStates = 8;
    int N = x_i[0];  // number of "patients"

    std::vector<scalar> dxdt(N * nStates);

    for (int i = 0; i < N; i++) {
      int k = nParms * i, j = nStates *  i;

      // PK variables
      CL = parms[k];
      Q = parms[k + 1];
      VC = parms[k + 2];
      VP = parms[k + 3];
      ka = parms[k + 4];
      k10 = CL / VC;
      k12 = Q / VC;
      k21 = Q / VP;

      // PD variables
      MTT = parms[k + 5];
      circ0 = parms[k + 6];
      alpha = parms[k + 7];
      gamma = parms[k + 8];
      ktr = 4 / MTT;
      prol = x[j + 3] + circ0,
      transit1 = x[j + 4] + circ0,
      transit2 = x[j + 5] + circ0,
      transit3 = x[j + 6] + circ0,
      circ = x[j + 7] + circ0;

      dxdt[j] = -ka * x[j];
      dxdt[j + 1] = ka * x[j] - (k10 + k12) * x[j + 1] + k21 * x[j + 2];
      dxdt[j + 2] = k12 * x[j + 1] - k21 * x[j + 2];

      conc = x[j + 1] / VC;
      Edrug = alpha * conc;

      dxdt[j + 3] = ktr * prol * ((1 - Edrug) * pow((circ0 / circ), gamma) - 1);
      dxdt[j + 4] = ktr * (prol - transit1);
      dxdt[j + 5] = ktr * (transit1 - transit2);
      dxdt[j + 6] = ktr * (transit2 - transit3);
      dxdt[j + 7] = ktr * (transit3 - circ);
    }

    return dxdt;
  }
};

struct run_num_pm {
  int N_;

  explicit run_num_pm(int N) : N_(N) { }

  template <typename T>
  inline
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  operator() (Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
    using std::vector;
    using stan::math::to_array_1d;
    using stan::math::integrate_ode_rk45;

    // fixed data argument.
    int nOde = 8;
    int nParms = 9;

    vector<T> init(N_ * nOde);
    for (int j = 0; j < N_; j++)
      for (int i = 0; i < nOde; i++)
        init[j * nOde + i] = x(j * (nParms + nOde) + nParms + i);

    vector<T> theta(N_ * nParms);
    for (int j = 0; j < N_; j++)
      for (int i = 0; i < nParms; i++)
        theta[j * nParms + i] = x(j * (nOde + nParms) + i);

    double t0 = 0.75;
    double t = 1.0;
    vector<double> dt(1, t - t0);
    vector<double> x_r(0);
    vector<int> x_i(1, N_);

    return to_vector(integrate_ode_rk45(FK_functor(), init, 0, dt, theta,
                                        x_r, x_i)[0]);
  }
};

struct FK_mix_functor {
  /**
   * parms contains both the PK and the PD parameters, and the
   * PK states.
   * x contains the PD states.
   */
  template <typename T0, typename T1, typename T2, typename T3>
  inline
  std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
  operator()(const T0& t,
             const std::vector<T1>& x,
             const std::vector<T2>& parms,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream__) const {
    using std::vector;
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using stan::math::to_vector;  // test
    typedef typename boost::math::tools::promote_args<T0, T1, T2, T3>::type scalar;
    int
      nPK = 3,
      nParmsPK = 5,
      nPD = 5,
      N = x_i[0],
      nParms = 9;

    // PK variables
    scalar VC;
    vector<scalar> parmsPK(nParmsPK);
    vector<scalar> initPK(nPK);

    // PD variables
    scalar MTT, circ0, alpha, gamma, ktr,
      prol, transit1, transit2, transit3, circ;

    vector<double> rate0(nPK, 0);  // no rates
    vector<scalar> predPK(nPK);
    scalar conc, Edrug;

    vector<scalar> dxdt(N * nPD);

    for (int i = 0; i < N; i++) {
      int k = (nParms + nPK) * i, j = nPD * i;

      // PK parameters
      VC = parms[k + 2];
      for (size_t i = 0; i < parmsPK.size(); i++)
        parmsPK[i] = parms[k + i];

      // PD variables
      MTT = parms[k + 5];
      circ0 = parms[k + 6];
      alpha = parms[k + 7];
      gamma = parms[k + 8];
      ktr = 4 / MTT;

      initPK[0] = parms[k + nParms];
      initPK[1] = parms[k + nParms + 1];
      initPK[2] = parms[k + nParms + 2];
      predPK = fTwoCpt(t, parmsPK, initPK, rate0);
      conc = predPK[1] / VC;
      Edrug = alpha * conc;

      prol = x[j] + circ0;
      transit1 = x[j + 1] + circ0;
      transit2 = x[j + 2] + circ0;
      transit3 = x[j + 3] + circ0;
      circ = x[j + 4] + circ0;

      // return object
      dxdt[j] = ktr * prol * ((1 - Edrug) * (pow((circ0 / circ), gamma)) - 1);
      dxdt[j + 1] = ktr * (prol - transit1);
      dxdt[j + 2] = ktr * (transit1 - transit2);
      dxdt[j + 3] = ktr * (transit2 - transit3);
      dxdt[j + 4] = ktr * (transit3 - circ);
    }

    return dxdt;
   }
};

struct run_mix_pm {
  int N_;

  explicit run_mix_pm(int N) : N_(N) { }

  template <typename T>
  inline
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  operator() (Eigen::Matrix<T, Eigen::Dynamic, 1> x) const {
    using std::vector;
    using stan::math::to_array_1d;
    using stan::math::to_vector;
    using stan::math::integrate_ode_rk45;

   // fixed data
   int nOde = 8, nParms = 9;
   vector<T> init(N_ * nOde);
   for (int i = 0; i < N_; i++)
     for (int j = 0; j < nOde; j++)
       init[i * nOde + j] = x(i * (nParms + nOde) + nParms + j);

   double t0 = 0.75, t = 1;
   vector<double> dt(1, t - t0);
   vector<double> x_r;
   vector<int> x_i(1, N_);  // store the number of "patients" in x_i.

   int nBase = 3, nStates = 5, nParmsPK = 5;

   vector<T> theta(N_ * (nBase + nParms));

   // Construct theta array
   for (int i = 0; i < N_; i++) {
     int j = (nParms + nBase) * i;
     for (int l = 0; l < nParms; l++)  // pass the ODE parameters
       theta[j + l] = x(i * (nParms + nOde) + l);
     for (int l = 0; l < nBase; l++)  // pass the base initial PK
       theta[j + nParms + l] = init[i * nOde + l];
   }

   // get initial states for num subsystem
   vector<T> init_num(N_ * nStates);
   for (int j = 0; j < N_; j++)
      for (int i = 0; i < nStates; i++)
        init_num[j * nStates + i] = init[j * nOde +  nBase + i];

   // solve numerical subsystem
    vector<T>
      temp_num = integrate_ode_rk45(FK_mix_functor(), init_num, 0, dt,
                                    theta, x_r, x_i)[0];

   // store numerical solution in return object
   Eigen::Matrix<T, Eigen::Dynamic, 1> temp_mix(N_ * (nBase + nStates));
   for (int j = 0; j < N_; j++)
     for (int i = 0; i < nStates; i++)
       temp_mix(nOde * j + nBase + i) = temp_num[nStates * j + i];

   // solve analytical subsystem
   vector<T> temp_an(N_ * nBase);
   vector<T> pred1_two(nBase);
   vector<double> rate(nBase, 0);
   vector<T> parmsPK(N_ * nParmsPK);
   vector<T> initPK(nBase);

    for (int i = 0; i < N_; i++) {
      int j = (nParms + nOde) * i;  // metric in x

      parmsPK[0] = x(j);  // CL
      parmsPK[1] = x(j + 1);  // Q
      parmsPK[2] = x(j + 2);  // VC
      parmsPK[3] = x(j + 3);  // VP
      parmsPK[4] = x(j + 4);  // ka

      initPK[0] = x(j + nParms);
      initPK[1] = x(j + nParms + 1);
      initPK[2] = x(j + nParms + 2);
      pred1_two = fTwoCpt(dt[0], parmsPK, initPK, rate);

      // store analytical solution in return object
      for (int j = 0; j < nBase; j++)
        temp_mix(nOde * i + j) = pred1_two[j];
    }

    return temp_mix;
  }
};


TEST(mix_solver, FK_vv)  {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using Eigen::VectorXd;
  using stan::math::var;
  using stan::math::jacobian;

  VectorXd fx_num, fx_mix;
  Matrix<double, Dynamic, Dynamic> J_num, J_mix;

  // Save CPU times in output file
  std::ofstream myfile;
  myfile.open("test/unit/math/torsten/mixSolver/mixSolverResult_pop_vv.csv");

  int NSim = 100;  // number of simulations
  for (int i = 0; i < NSim; i++) myfile << i << ", ";
    myfile << "\n";

  int nParms = 9, nOde = 8, N_max = 30;

  for (int N = 1; N <= N_max; N++) {
    std::cout << (N - 1) << " / " << N_max << std::endl;

    // ODE parameters
    Matrix<double, Dynamic, 1> x(N * (nParms + nOde));
    for (int i = 0; i < N; i ++) {
      int j = i * (nParms + nOde);

      // PK parameters
      x(j) = 10; // CL
      x(j + 1) = 15;  // Q
      x(j + 2) = 35;  // VC
      x(j + 3) = 105;  // VP
      x(j + 4) = 2;  // ka
      x(j + 5) = 125;  // MTT
      x(j + 6) = 5;  // Circ0
      x(j + 7) = 3e-4;  // alpha
      x(j + 8) = 0.17;  // gamma

      // initial states
      x(j + nParms) = 1.785041e+04;
      x(j + nParms + 1) = 45426.123;
      x(j + nParms + 2) = 9867.0965;
      x(j + nParms + 3) = -0.032802493;
      x(j + nParms + 4) = -3.054236e-04;
      x(j + nParms + 5) = -2.003849e-06;
      x(j + nParms + 6) = -1.020732e-08;
      x(j + nParms + 7) = -4.260406e-11;
    }

    // Results from mrgsolve - use that to check solution
    VectorXd fx_mrgSolve(N * nOde);
    for (int i = 0; i < N; i ++) {
      fx_mrgSolve(nOde * i) = 1.082682e+04;
      fx_mrgSolve(nOde * i + 1) = 44779.912;
      fx_mrgSolve(nOde * i + 2) = 14296.1069;
      fx_mrgSolve(nOde * i + 3) = -0.048232219;
      fx_mrgSolve(nOde * i + 4) = -6.261043e-04;
      fx_mrgSolve(nOde * i + 5) = -5.620476e-06;
      fx_mrgSolve(nOde * i + 6) = -3.883612e-08;
      fx_mrgSolve(nOde * i + 7) = -2.188286e-10;
    }

    myfile << N << "a,";

    // Full numerical method
    for (int i = 0; i < NSim; i++) {
      clock_t start = clock();
      jacobian(run_num_pm(N), x, fx_num, J_num);
      clock_t end = clock();

      myfile << (float)(end - start) / CLOCKS_PER_SEC << ", ";

      // Check solution
      expect_matrix_eq(fx_mrgSolve, fx_num);
    }
    myfile << "\n" << N << "b,";

    // Mix Method
    for (int i = 0; i < NSim; i++) {
      clock_t start = clock();
      jacobian(run_mix_pm(N), x, fx_mix, J_mix);
      clock_t end = clock();

      // std::cout << fx_mix << std::endl;

      myfile << (float)(end - start) / CLOCKS_PER_SEC << ", ";

      // Check solution.
      // The relative error is determined empirically. If no
      // relative error is required, use expect_matrix_eq which
      // compares result within floating precision. Note mrgsolve
      // also approximates the solution numerically.
      expect_near_matrix_eq(fx_mrgSolve, fx_mix, 1e-6);
      // std::cout << fx_mix << std::endl;
    }
    myfile << "\n";
  }

  myfile.close();

  // Compare the Jacobians produced by both methods (note: only compares last
  // two simulations)
  expect_near_matrix_eq(J_num, J_mix, 1e-5, 1e-10);
}
