#ifndef STAN_MATH_TORSTEN_REFACTOR_INTEGRATOR_HPP
#define STAN_MATH_TORSTEN_REFACTOR_INTEGRATOR_HPP

#include <Eigen/Dense>
#include <stan/math/prim/arr/functor/integrate_ode_rk45.hpp>
#include <stan/math/rev/mat/functor/integrate_ode_bdf.hpp>
#include <iostream>
#include <string>
#include <vector>

namespace torsten {

  /** 
   * A vector of std::function(). The members are various
   * integrators. The types of integrators are determined at
   * compile time by template parameter Intg, the
   * index pointing
   * at according vector entry
   */
  template<typename F, typename T1, typename T2>
  struct IntegratorList {
    static const std::vector<std::function<
      std::vector<std::vector<typename stan::return_type<T1, T2>::type> >
      (const F& f,
       const std::vector<T1> y0,
       double t0,
       const std::vector<double>& ts,
       const std::vector<T2>& theta,
       const std::vector<double>& x,
       const std::vector<int>& x_int,
       std::ostream* msgs,
       double rel_tol,
       double abs_tol,
       long int max_num_steps)> > integrators;
  };

  template<typename F, typename T1, typename T2>
  const std::vector<std::function<
    std::vector<std::vector<typename stan::return_type<T1, T2>::type> >
      (const F& f,
       const std::vector<T1> y0,
       double t0,
       const std::vector<double>& ts,
       const std::vector<T2>& theta,
       const std::vector<double>& x,
       const std::vector<int>& x_int,
       std::ostream* msgs,
       double rel_tol,
       double abs_tol,
       long int max_num_steps)> > IntegratorList<F, T1, T2>::integrators =
    {
      stan::math::integrate_ode_rk45<F, T1, T2>,
      stan::math::integrate_ode_bdf<F, T1, T2>
    };

/**
 *  Construct functors that run the ODE integrator. Specify integrator
 *  type, the base ODE system, and the tuning parameters (relative tolerance,
 *  absolute tolerance, and maximum number of steps).
 */
struct TorstenIntegrator {
private:
  double rel_tol, abs_tol;
  long int max_num_steps;  // NOLINT(runtime/int)
  std::ostream* msgs;
  const int solver_type;

public:
  static const int RK45;
  static const int BDF;

  TorstenIntegrator() : 
    rel_tol       (1e-10),
    abs_tol       (1e-10),
    max_num_steps (1e8),
    msgs          (0),
    solver_type   (0)
  {}

  TorstenIntegrator(double p_rel_tol,
                    double p_abs_tol,
                    long int p_max_num_steps,
                    std::ostream* p_msgs,
                    int p_solver_type) :
    rel_tol       (p_rel_tol),
    abs_tol       (p_abs_tol),
    max_num_steps (p_max_num_steps),
    msgs          (p_msgs),
    solver_type   (p_solver_type)
  {}

  TorstenIntegrator(const TorstenIntegrator& other) :
      rel_tol       (other.rel_tol),
      abs_tol       (other.abs_tol),
      max_num_steps (other.max_num_steps),
      msgs          (other.msgs),
      solver_type   (other.solver_type)
  {}

  // CONSTRUCTOR FOR OPERATOR
  template<typename F, typename T1, typename T2>
  std::vector<std::vector<typename stan::return_type<T1, T2>::type> >
  operator() (const F& f,
              const std::vector<T1> y0,
              const double t0,
              const std::vector<double>& ts,
              const std::vector<T2>& theta,
              const std::vector<double>& x,
              const std::vector<int>& x_int) const {
    return IntegratorList<F, T1, T2>::integrators.
      at(solver_type)(f, y0, t0, ts, theta, x, x_int,
                      msgs,
                      rel_tol,
                      abs_tol,
                      max_num_steps);
  }
};
  
  const int TorstenIntegrator::RK45 = 0;
  const int TorstenIntegrator::BDF  = 1;
}

#endif
