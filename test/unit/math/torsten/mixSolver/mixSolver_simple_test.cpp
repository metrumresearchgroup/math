#include <stan/math/rev/mat.hpp>  // FIX ME - more specific
#include <test/unit/math/prim/mat/fun/expect_matrix_eq.hpp>
#include <gtest/gtest.h>
#include <ostream>
#include <fstream>
#include <vector>


struct OneCpt_functor {
  /**
   * parms contains all parameters.
   * x contains all initial states.
   */
  template <typename T0, typename T1, typename T2, typename T3>
  inline
  std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
  operator()(const T0& t,
             const std::vector<T1>& y,
             const std::vector<T2>& parms,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream__) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    typedef typename boost::math::tools::promote_args<T0, T1, T2, T3>::type scalar;
    T2
      CL = parms[0],
      VC = parms[1],
      ka = parms[2];

    int nStates = 2;
    int N = x_i[0];  // number of "patients"

    std::vector<scalar> dydt(N * nStates);

    for (int i = 0; i < (N * nStates - 1); i += 2) {
      dydt[i] = -ka * y[i];
      dydt[i + 1] = ka * y[i] - CL / VC * y[i + 1];
    }

    return dydt;
  }
};

struct OneCpt_mix_functor {
  /**
   * parms contains all parameters and the y1 initial state.
   * x contains the y2 initial state.
   */
  template <typename T0, typename T1, typename T2, typename T3>
  inline
  std::vector<typename boost::math::tools::promote_args<T0, T1, T2, T3>::type>
  operator()(const T0& t,
             const std::vector<T1>& y,
             const std::vector<T2>& parms,
             const std::vector<T3>& x_r,
             const std::vector<int>& x_i,
             std::ostream* pstream__) const {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    typedef typename boost::math::tools::promote_args<T0, T1, T2, T3>::type scalar;
    T2
      CL = parms[0],
      VC = parms[1],
      ka = parms[2];

    int N = x_i[0];
    int nOde = 1;
    std::vector<scalar> dydt(N * nOde);

    std::vector<scalar> y_f(N);
    for (int i = 0; i < N; i++) {
      y_f[i] = x_r[i] * exp(-ka * t);
      dydt[i] = ka * y_f[i] - CL / VC * y[i];
    }

    return dydt;
  }
};

struct run_num {

  template <typename T>
  inline
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  operator() (Eigen::Matrix<T, Eigen::Dynamic, 1> x) const {
    using std::vector;
    using stan::math::to_array_1d;
    using stan::math::integrate_ode_rk45;

    // fixed data argument. DEV - compute these outside run_num?
    int N = 100;  // number of "patients"
    int nOde = 2;
    vector<double> init(N * nOde);
    for (int i = 0; i < (N * nOde - 1); i += 2) {
      init[i] = 1000;  // bolus dose
      init[i + 1] = 0;
    }
    double t0 = 0;
    double t = 0.25;
    vector<double> dt(1, t - t0);
    vector<double> x_r(0);
    vector<int> x_i(1, N);

    vector<T> theta = to_array_1d(x);

    return to_vector(integrate_ode_rk45(OneCpt_functor(), init, 0, dt,
                                        theta, x_r, x_i)[0]);
  }
};

struct run_mix {

  template <typename T>
  inline
  Eigen::Matrix<T, Eigen::Dynamic, 1>
  operator() (Eigen::Matrix<T, Eigen::Dynamic, 1> x) const {
    using std::vector;
    using stan::math::to_array_1d;
    using stan::math::integrate_ode_rk45;

    // fixed data argument. DEV - compute these outside run_num?
    int N = 100;  // number of "patients"
    int nOde = 2;  // total number of ODEs per patient
    vector<double> init(N * nOde);
    for (int i = 0; i < (N * nOde - 1); i += 2) {
      init[i] = 1000;  // bolus dose
      init[i + 1] = 0;
    }
    double t0 = 0;
    double t = 0.25;
    vector<double> dt(1, t - t0);
    vector<double> x_r(0);
    vector<int> x_i(1, N);

    int nBase = 1, nStates = 1;

    vector<T> theta = to_array_1d(x);

    // The initial states are passed as data
    // Augment x_r.
    for (int j = 0; j < N; j++)
      for (int i = 0; i < nBase; i++)
        x_r.push_back(init[j * nOde + i]);

    // get initial states for num subsystem
    vector<double> init_num(N * nStates);
    for (int i = 0; i < N; i++)
      for (int j = 0; j < nStates; j++) {
        init_num[i] = init[i * nOde +  nBase + j];
      }

    // solve numerical subsystems
    vector<T>
      temp_num = integrate_ode_rk45(OneCpt_mix_functor(), init_num, 0, dt,
                                    theta, x_r, x_i)[0];

    // solve analytical subsystems
    vector<T> temp_an(N);

    for (int i = 0; i < N; i++)
      temp_an[i] = init[2 * i] * exp(-x[2] * dt[0]);

    Eigen::Matrix<T, Eigen::Dynamic, 1> temp_mix(N * (nBase + nStates));
    for (int j = 0; j < N; j++) {
      for (int i = 0; i < nBase; i++) temp_mix(nOde * j + i) = temp_an[nBase * j + i];
      for (int i = 0; i < nStates; i++) temp_mix(nOde * j + nBase + i) = temp_num[nStates * j + i];
    }

    return temp_mix;
  }
};

TEST(mix_solver, OneCpt) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::var;
  using stan::math::jacobian;

  Matrix<double, Dynamic, 1> x(3);
  x[0] = 10;  // CL
  x[1] = 80;  // VC
  x[2] = 1.2;  // ka

  Matrix<double, Dynamic, 1> fx_num, fx_mix;
  Matrix<double, Dynamic, Dynamic> J_num, J_mix;

  // Save CPU times inside output file
  std::ofstream myfile;
  myfile.open("test/unit/math/torsten/mixSolver/mixSolverResult_simple.csv");

  int NSim = 1000;  // number of simulations
  for (int i = 0; i < NSim; i++) myfile << i << ", ";
  myfile << "0\n";

  // Full numerical method
  for (int i = 0; i < NSim; i++) {
    clock_t start = clock();
    jacobian(run_num(), x, fx_num, J_num);
    clock_t end = clock();

    myfile << (float)(end - start) / CLOCKS_PER_SEC << ", ";
  }
  myfile << "0\n";

  // Mix method
  for (int i = 0; i < NSim; i++) {
    clock_t start = clock();
    jacobian(run_mix(), x, fx_mix, J_mix);
    clock_t end = clock();

    myfile << (float)(end - start) / CLOCKS_PER_SEC << ", ";
  }
  myfile << "0\n";
  myfile.close();

  // make sure both method compute the same result
  expect_matrix_eq(fx_num, fx_mix);
  expect_matrix_eq(J_num, J_mix);
}
