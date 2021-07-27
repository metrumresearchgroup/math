#include <gtest/gtest.h>
#include <stan/math/torsten/dsolve/sundials_check.hpp>
#include <arkode/arkode_erkstep.h>
#include <arkode/arkode_butcher_erk.h>
#include <nvector/nvector_serial.h>
#include <boost/numeric/odeint.hpp>
#include <boost/numeric/odeint/external/eigen/eigen.hpp>
#include <vector>
#include <iostream>
#include <Eigen/Dense>

template<typename Ode>
struct arkode_rhs {
  static int fn(double t, N_Vector y, N_Vector ydot, void* user_data) {
    Ode* ode = static_cast<Ode*>(user_data);
    (*ode)(t, y, ydot);
    return 0;
  }  
};

template<typename Ode>
struct arkode_wt {
  static int fn(N_Vector y, N_Vector ewt, void* user_data) {
    Ode* ode = static_cast<Ode*>(user_data);
    int n = NV_LENGTH_S(y);
    for (int i = 0; i < n; ++i) {
      NV_Ith_S(ewt, i) = 1.0/(ode -> atol + ode -> rtol * abs(NV_Ith_S(y, i)));
    }
    return 0;
  }
};

/**
 * RAII for arkode
 */
template <typename Ode>
struct PMXOdeService {
  int n;                        /**< ode system size */
  N_Vector nv_y;                /**< state vector */
  void* mem;                    /**< memory */

  /**
   * Construct  mem & workspace
   *
   */
  PMXOdeService(std::vector<double> const& y0) :
    n(y0.size()),
    nv_y(N_VNew_Serial(n)),
    mem(ERKStepCreate(arkode_rhs<Ode>::fn, 0.0, nv_y)) {
    for (auto i = 0; i < n; ++i) {
      NV_Ith_S(nv_y, i) = y0[i];
    }
  }

  ~PMXOdeService() {
    ERKStepFree(&mem);          // Free integrator memory
    N_VDestroy(nv_y);           // Free y vector
  }
};

struct ode_observer {
  int n;
  int nt;
  int step_counter_;
  Eigen::MatrixXd result;

  ode_observer(int n0, int nt0) :
    n(n0), nt(nt0), step_counter_(0),
    result(n0, nt0)
  {}

  void operator()(const std::vector<double>& curr_result, double t) {
    if(t > 0) {
      for (auto i = 0; i < n; ++i) {
        result(i, step_counter_) = curr_result[i];
      }
      step_counter_++;
    }
  }

  void operator()(const N_Vector& curr_result, double t) {
    if(t > 0) {
      for (auto i = 0; i < n; ++i) {
        result(i, step_counter_) = NV_Ith_S(curr_result, i);
      }
      step_counter_++;
    }
  }
};

struct lotka_volterra {
  static constexpr double rtol=1.e-12;
  static constexpr double atol=1.e-12;
  static constexpr int max_num_steps=1000000;

  const std::vector<double>& theta;
  long int n_eval;

  lotka_volterra(const std::vector<double>& theta0) :
    theta(theta0), n_eval(0)
  {}

  void operator()(double t_in, N_Vector& y, N_Vector& ydot) {
    n_eval++;
    double alpha = theta[0];
    double beta = theta[1];
    double gamma = theta[2];
    double delta = theta[3];

    NV_Ith_S(ydot, 0) = (alpha - beta * NV_Ith_S(y, 1)) * NV_Ith_S(y, 0);
    NV_Ith_S(ydot, 1) = (-gamma + delta * NV_Ith_S(y, 0)) * NV_Ith_S(y, 1);
  }

  void operator()(const std::vector<double>& y, std::vector<double>& ydot, double t) {
    n_eval++;
    double alpha = theta[0];
    double beta = theta[1];
    double gamma = theta[2];
    double delta = theta[3];

    ydot[0] = (alpha - beta * y[1]) * y[0];
    ydot[1] = (-gamma + delta * y[0]) * y[1];
  }
};

TEST(arkode, lotka) {
  int n = 2;
  std::vector<double> theta{1.5, 1.05, 1.5, 2.05};
  std::vector<double> y0{0.3, 0.8};
  lotka_volterra ode(theta);
  PMXOdeService<lotka_volterra> serv(y0);
  void* mem = serv.mem;
  N_Vector& y = serv.nv_y;
  int nt = 10;
  std::vector<double> ts(nt);
  for (auto i = 0; i < nt; ++i) {
    ts[i] = (i + 1) * 1000.0;
  }

  CHECK_SUNDIALS_CALL(ERKStepReInit(mem, arkode_rhs<lotka_volterra>::fn, 0.0, y));
  CHECK_SUNDIALS_CALL(ERKStepSetUserData(mem, static_cast<void*>(&ode)));
  CHECK_SUNDIALS_CALL(ERKStepSStolerances(mem, ode.rtol, ode.atol));
  CHECK_SUNDIALS_CALL(ERKStepSetMaxNumSteps(mem, ode.max_num_steps));
  CHECK_SUNDIALS_CALL(ERKStepSetAdaptivityMethod(mem, 2, SUNTRUE, SUNFALSE, NULL));

  CHECK_SUNDIALS_CALL(ERKStepSetTableNum(mem, DORMAND_PRINCE_7_4_5));
  ERKStepSetInitStep(mem, 0.1);

  double t0 = 0;
  double t1 = t0;
      
  ode_observer ob(n, ts.size());
  for (auto i = 0; i < ts.size(); ++i) {
    CHECK_SUNDIALS_CALL(ERKStepEvolve(mem, ts[i], y, &t1, ARK_NORMAL));
    ob(y, t1);
  }

  long int n_eval;
  ERKStepGetNumRhsEvals(mem, &n_eval);
  assert(ode.n_eval == n_eval);
  std::cout << "# of RHS evals: " << ode.n_eval << "\n";
}

// TEST(arkode_wmrs, lotka) {
//   int n = 2;
//   std::vector<double> theta{1.5, 1.05, 1.5, 2.05};
//   std::vector<double> y0{0.3, 0.8};
//   lotka_volterra ode(theta);
//   PMXOdeService<lotka_volterra> serv(y0);
//   void* mem = serv.mem;
//   N_Vector& y = serv.nv_y;
//   int nt = 10;
//   std::vector<double> ts(nt);
//   for (auto i = 0; i < nt; ++i) {
//     ts[i] = (i + 1) * 1000.0;
//   }

//   CHECK_SUNDIALS_CALL(ERKStepReInit(mem, arkode_rhs<lotka_volterra>::fn, 0.0, y));
//   CHECK_SUNDIALS_CALL(ERKStepSetUserData(mem, static_cast<void*>(&ode)));
//   CHECK_SUNDIALS_CALL(ERKStepSStolerances(mem, ode.rtol, ode.atol));
//   CHECK_SUNDIALS_CALL(ERKStepSetMaxNumSteps(mem, ode.max_num_steps));
//   CHECK_SUNDIALS_CALL(ERKStepSetAdaptivityMethod(mem, 2, SUNTRUE, SUNFALSE, NULL));

//   CHECK_SUNDIALS_CALL(ERKStepSetTableNum(mem, DORMAND_PRINCE_7_4_5));
//   ERKStepWFtolerances(mem, arkode_wt<lotka_volterra>::fn);
//   ERKStepSetInitStep(mem, 0.1);

//   double t0 = 0;
//   double t1 = t0;
      
//   ode_observer ob(n, ts.size());
//   for (auto i = 0; i < ts.size(); ++i) {
//     CHECK_SUNDIALS_CALL(ERKStepEvolve(mem, ts[i], y, &t1, ARK_NORMAL));
//     ob(y, t1);
//   }
//   std::cout << "taki test: " << ode.n_eval << "\n";
//   // std::cout << ob.result << "\n";
// }

TEST(odeint, lotka) {
  int n = 2;
  std::vector<double> theta{1.5, 1.05, 1.5, 2.05};
  std::vector<double> y0{0.3, 0.8};
  lotka_volterra ode(theta);

  double t0 = 0;
  int nt = 10;
  std::vector<double> ts(nt + 1);
  ts[0] = t0;
  for (auto i = 1; i < nt + 1; ++i) {
    ts[i] = i * 1000.0;
  }

  const double init_dt = 0.1;
  ode_observer ob(n, ts.size() - 1);
  integrate_times(make_dense_output(ode.atol, ode.rtol, boost::numeric::odeint::runge_kutta_dopri5<std::vector<double>, double, std::vector<double>, double>()),
                  boost::ref(ode), y0,
                  ts.begin(), ts.end(),
                  init_dt, boost::ref(ob),
                  boost::numeric::odeint::max_step_checker(ode.max_num_steps));

  std::cout << "# of RHS evals: " << ode.n_eval << "\n";
}
