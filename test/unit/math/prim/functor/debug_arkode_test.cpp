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

  void print_n_eval(std::string info) {
    std::cout << "# of RHS evals ( " << info << " ) :" << n_eval << "\n";
  }

  void operator()(double t_in, N_Vector& y, N_Vector& ydot) {
    n_eval++;
    double alpha = theta[0];
    double beta = theta[1];
    double gamma = theta[2];
    double delta = theta[3];

    // NV_Ith_S(ydot, 0) = (alpha - beta * NV_Ith_S(y, 1)) * NV_Ith_S(y, 0);
    // NV_Ith_S(ydot, 1) = (-gamma + delta * NV_Ith_S(y, 0)) * NV_Ith_S(y, 1);
    double *ydata = NV_DATA_S(y);
    double *yddata = NV_DATA_S(ydot);    
    yddata[0] = (alpha - beta * ydata[1]) * ydata[0];
    yddata[1] = (-gamma + delta * ydata[0]) * ydata[1];
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
  std::vector<double> theta{1.5, 1.05, 1.5, 2.05};
  std::vector<double> y0{0.3, 0.8};
  int n = y0.size();
  lotka_volterra ode(theta);
  
  int nt = 10;
  std::vector<double> ts(nt);
  for (auto i = 0; i < nt; ++i) {
    ts[i] = (i + 1) * 1000.0;
  }

  N_Vector y = N_VNew_Serial(n);
  NV_Ith_S(y, 0) = y0[0];
  NV_Ith_S(y, 1) = y0[1];
  void* mem = ERKStepCreate(arkode_rhs<lotka_volterra>::fn, 0.0, y);
  CHECK_SUNDIALS_CALL(ERKStepSetUserData(mem, static_cast<void*>(&ode)));
  CHECK_SUNDIALS_CALL(ERKStepSStolerances(mem, ode.rtol, ode.atol));
  CHECK_SUNDIALS_CALL(ERKStepSetMaxNumSteps(mem, ode.max_num_steps));
  CHECK_SUNDIALS_CALL(ERKStepSetAdaptivityMethod(mem, 2, SUNTRUE, SUNFALSE, NULL));

  CHECK_SUNDIALS_CALL(ERKStepSetTableNum(mem, DORMAND_PRINCE_7_4_5));
  ERKStepSetInitStep(mem, 0.1);

  double t0 = 0;
  double t1 = t0;
      
  ode_observer ob(n, ts.size());


  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double, std::milli> elapsed;
  start = std::chrono::system_clock::now();
  for (auto i = 0; i < ts.size(); ++i) {
    CHECK_SUNDIALS_CALL(ERKStepEvolve(mem, ts[i], y, &t1, ARK_NORMAL));
    ob(y, t1);
  }
  end = std::chrono::system_clock::now();
  elapsed = (end - start);

  std::cout << "ERKStep elapsed time: " << elapsed.count() << " ms\n";

  long int n_eval;
  ERKStepGetNumRhsEvals(mem, &n_eval);
  assert(ode.n_eval == n_eval);
  ode.print_n_eval("arkode");

  ERKStepFree(&mem);          // Free integrator memory
  N_VDestroy(y);           // Free y vector

}

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

  ode.print_n_eval("odeint");
}

TEST(arkode, rhs_eval) {
  int n = 2;
  std::vector<double> theta{1.5, 1.05, 1.5, 2.05};
  std::vector<double> y0{0.3, 0.8};
  lotka_volterra ode(theta);

  const long int n_eval = 99999999;

  N_Vector y = N_VNew_Serial(n);
  N_Vector ydot = N_VNew_Serial(n);
  NV_Ith_S(y, 0) = y0[0];         // dummy
  NV_Ith_S(y, 1) = y0[1];         // dummy

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double, std::milli> elapsed;
  start = std::chrono::system_clock::now();
  for (auto i = 0; i < n_eval; ++i) {
    ode(0.1 * i, y, ydot);    
  }
  end = std::chrono::system_clock::now();
  elapsed = (end - start);

  std::cout << "arkode RHS elapsed time: " << elapsed.count() << " ms\n";
}

TEST(odeint, rhs_eval) {
  int n = 2;
  std::vector<double> theta{1.5, 1.05, 1.5, 2.05};
  std::vector<double> y{0.3, 0.8}, ydot(n);
  lotka_volterra ode(theta);

  const long int n_eval = 99999999;

  std::chrono::time_point<std::chrono::system_clock> start, end;
  std::chrono::duration<double, std::milli> elapsed;
  start = std::chrono::system_clock::now();
  for (long int i = 0; i < n_eval; ++i) {
    ode(y, ydot, 0.1 * i);
  }
  end = std::chrono::system_clock::now();
  elapsed = (end - start);

  std::cout << "odeint RHS elapsed time: " << elapsed.count() << " ms\n";
}
