#include <stan/math/rev/mat.hpp>
#include <stan/math/torsten/PKModel/Pred/fTwoCpt.hpp>
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
    // PK variables
    scalar
      CL = parms[0],
      Q = parms[1],
      VC = parms[2],
      VP = parms[3],
      ka = parms[4],
      k10 = CL / VC,
      k12 = Q / VC,
      k21 = Q / VP;

    // PD variables
    scalar
      MTT = parms[5],
      circ0 = parms[6],
      alpha = parms[7],
      gamma = parms[8],
      ktr = 4 / MTT;

    int nStates = 8;
    int N = x_i[0];  // number of "patients"

    std::vector<scalar> dxdt(N * nStates);

    for (int i = 0; i < (N * nStates - 1); i += nStates) {
      scalar
        prol = x[i + 3] + circ0,
        transit1 = x[i + 4] + circ0,
        transit2 = x[i + 5] + circ0,
        transit3 = x[i + 6] + circ0,
        circ = x[i + 7] + circ0;

     dxdt[i] = -ka * x[i];
     dxdt[i + 1] = ka * x[i] - (k10 + k12) * x[i + 1] + k21 * x[i + 2];
     dxdt[i + 2] = k12 * x[i + 1] - k21 * x[i + 2];

     scalar conc = x[i + 1] / VC;
     scalar Edrug = alpha * conc;

      dxdt[i + 3] = ktr * prol * ((1 - Edrug) * pow((circ0 / circ), gamma) - 1);
      dxdt[i + 4] = ktr * (prol - transit1);
      dxdt[i + 5] = ktr * (transit1 - transit2);
      dxdt[i + 6] = ktr * (transit2 - transit3);
      dxdt[i + 7] = ktr * (transit3 - circ);
    }

    return dxdt;
  }
};

struct run_num_pm {
   // pm regime: parameters are var, but inits aren't.

  template <typename T>
  inline
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  operator() (Eigen::Matrix<T, Eigen::Dynamic, 1>& x) const {
    using std::vector;
    using stan::math::to_array_1d;
    using stan::math::integrate_ode_rk45;
    
    // fixed data argument.
    int N = 100;  // number of "patients"
    int nOde = 8;
    vector<double> init(N * nOde, 0);
    for (int i = 0; i < (N * nOde - 1); i += nOde)
      init[i] = 80000;  // bolus dose

    double t0 = 0;
    double t = 1.0;
    vector<double> dt(1, t - t0);
    vector<double> x_r(0);
    vector<int> x_i(1, N);
    
    vector<T> theta = to_array_1d(x);

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
    // PK variables
    scalar VC = parms[2];

    // PD variables
    scalar
      MTT = parms[5],
      circ0 = parms[6],
      alpha = parms[7],
      gamma = parms[8],
      ktr = 4 / MTT,
      prol, transit1, transit2, transit3, circ;

    int nPK = 3, nPD = 5, N = x_i[0];
    vector<scalar> parmsPK(5);
    for (size_t i = 0; i < parmsPK.size(); i++)
      parmsPK[i] = parms[i];

    vector<scalar> initPK(nPK);
    vector<double> rate0(nPK, 0);  // no rates
    scalar conc, Edrug;

    vector<scalar> dxdt(N * nPD);

    for (int i = 0; i < N; i++) {  
      initPK[0] = x_r[i * nPK];
      initPK[1] = x_r[i * nPK + 1];
      initPK[2] = x_r[i * nPK + 2];
      conc = fTwoCpt(t, parmsPK, initPK, rate0)[1] / VC;
      Edrug = alpha * conc;

      prol = x[i * nPD] + circ0;
      transit1 = x[i * nPD + 1] + circ0;
      transit2 = x[i * nPD + 2] + circ0;
      transit3 = x[i * nPD + 3] + circ0;
      circ = x[i * nPD + 4] + circ0;

      // return object
      dxdt[i * nPD] = ktr * prol * ((1 - Edrug) * (pow((circ0 / circ), gamma)) - 1);
      dxdt[i * nPD + 1] = ktr * (prol - transit1);
      dxdt[i * nPD + 2] = ktr * (transit1 - transit2);
      dxdt[i * nPD + 3] = ktr * (transit2 - transit3);
      dxdt[i * nPD + 4] = ktr * (transit3 - circ);      
    }
    
    return dxdt;
   }
};

struct run_mix_pm {

  template <typename T>
  inline
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  operator() (Eigen::Matrix<T, Eigen::Dynamic, 1> x) const {
    using std::vector;
    using stan::math::to_array_1d;
    using stan::math::integrate_ode_rk45;

   // fixed data
   int N = 100;
   int nOde = 8;
   vector<double> init(N * nOde, 0);
   for (int i = 0; i < (N * nOde - 1); i += nOde) init[i] = 80000;

   double t0 = 0, t = 1;
   vector<double> dt(1, t - t0);

   int nBase = 3, nStates = 5;

   vector<T> theta = to_array_1d(x);
   
   // The base initial states are passed as data.
   vector<double> x_r(N * nBase);
   for (int j = 0; j < N; j++)
     for (int i = 0; i < nBase; i++)
       x_r[j * nBase + i] = init[j * nOde + i];

   // store the number of "patients" in x_i.
   vector<int> x_i(1, N);

   // get initial states for num subsystem
   vector<double> init_num(N * nStates);
   for (int i = 0; i < N; i++)
      for (int j = 0; j < nStates; j++)
        init_num[i] = init[i * nOde +  nBase + j];

   // solve numerical subsystem
    vector<T>
      temp_num = integrate_ode_rk45(FK_mix_functor(), init_num, 0, dt,
                                    theta, x_r, x_i)[0];   

   Eigen::Matrix<T, Eigen::Dynamic, 1> temp_mix(N * (nBase + nStates));
   // store numerical solution in return object
   for (int j = 0; j < N; j++)
     for (int i = 0; i < nStates; i++)
       temp_mix(nOde * j + nBase + i) = temp_num[nStates * j + i];

   // solve analytical subsystem
   vector<T> temp_an(N * nBase);
   vector<T> pred1_two(nBase);
   vector<double> rate(nBase, 0);
   vector<T> parmsPK(5);
   for (size_t i = 0; i < parmsPK.size(); i++) parmsPK[i] = x(i);
   vector<double> initPK(nBase);

    for (int i = 0; i < N; i++) {
      initPK[0] = x_r[i * nBase];
      initPK[1] = x_r[i * nBase + 1];
      initPK[2] = x_r[i * nBase + 2];
      pred1_two = fTwoCpt(dt[0], parmsPK, initPK, rate);
      
      // store analytical solution in return object
      for (int j = 0; j < nBase; j++)
        temp_mix(nOde * i + j) = pred1_two[j];
    }

    return temp_mix;
  }
};


TEST(mix_solver, FK)  {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using Eigen::VectorXd;
  using stan::math::var;
  using stan::math::jacobian;

  Matrix<double, Dynamic, 1> x(9);
  x(0) = 10;  // CL
  x(1) = 15;  // Q
  x(2) = 35;  // VC
  x(3) = 105;  // VP
  x(4) = 2;  // ka
  x(5) = 125;  // MTT
  x(6) = 5;  // Circ0
  x(7) = 3e-4;  // alpha
  x(8) = 0.17;  // gamma

  VectorXd fx_num, fx_mix;
  Matrix<double, Dynamic, Dynamic> J_num, J_mix;
  
  std::cout << fx_num << std::endl;

  jacobian(run_num_pm(), x, fx_num, J_num);
  jacobian(run_mix_pm(), x, fx_mix, J_mix);

  // std::cout << fx_mix << std::endl;

  // Check solution
  int N = 100;  // number of patients
  int nOde = 8;
  VectorXd fx_mrgSolve(N * nOde);
  for (int i = 0; i < N; i ++) {
    fx_mrgSolve(nOde * i) = 10826.82;
    fx_mrgSolve(nOde * i + 1) = 44779.91;
    fx_mrgSolve(nOde * i + 2) = 14296.11;
    fx_mrgSolve(nOde * i + 3) = -0.04823222;
    fx_mrgSolve(nOde * i + 4) = -0.0006261043;
    fx_mrgSolve(nOde * i + 5) = -5.620476e-06;
    fx_mrgSolve(nOde * i + 6) = -3.883612e-08;
    fx_mrgSolve(nOde * i + 7) = -2.188286e-10;
  }
  
  expect_matrix_eq(fx_mrgSolve, fx_num);
  expect_near_matrix_eq(fx_mrgSolve, fx_mix, 1e-6);  // why do I need an approx?
  
  // Compare the Jacobians produced by both methods.
  // expect_near_matrix_eq(J_num, J_mix, 1e-6);
  
  // FIX ME - determine the relative errors empirically
  
}
    
    
    
    

