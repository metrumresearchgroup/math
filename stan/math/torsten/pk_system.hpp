#ifndef PK_SYSTEM_HPP
#define PK_SYSTEM_HPP

// #include <stan/math/torsten/event_list.hpp>
// #include <stan/math/torsten/modelparameters2.hpp>
#include <stan/math/torsten/PKModel/Pred/PolyExp.hpp>

namespace refactor {

using boost::math::tools::promote_args;

// depend on model, we can have arbitrary number of
// parameters, e.g. biovar, k12, k10, ka. Each parameter
// can be data or var.

  // conversion constructor and indexing sugar
  template<typename T>
  struct PKParameterVectorx {
    std::vector<std::vector<T> > v;

    PKParameterVectorx(std::vector<std::vector<T> > v0) : v(v0) {}

    PKParameterVectorx(std::vector<T> v0) : v({1, v0}) {}

    std::vector<T>& operator[](const int index) {
      return v.size() == 1 ? v[0] : v[index];
    }

    const std::vector<T>& operator[](const int index) const {
      return v.size() == 1 ? v[0] : v[index];
    }
  };

template <typename T_time,
          typename T_amt,
          typename T_rate,
          typename T_ii,
          typename T_par,
          typename T_biovar,
          typename T_tlag>
struct PKSystem {
  torsten::EventHistory<typename promote_args<T_time, T_tlag>::type, T_amt, T_rate, T_ii>
  events;
  torsten::ModelParameterHistory<typename promote_args<T_time, T_tlag>::type, T_par, T_biovar, T_tlag>
  parameters;
  torsten::RateHistory<typename promote_args<T_time, T_tlag>::type, T_rate>
  rates;
  int nKeep_;
  
  // output type for the system
  typedef typename promote_args<T_time, T_amt, T_rate, T_ii,
                                typename promote_args<T_par, T_biovar, T_tlag>::type>::type
  scalar_type;
  typedef typename promote_args<T_time, T_tlag>::type T_tau;
  typedef typename promote_args<T_rate, T_biovar>::type T_rate2;

  // constructors
  PKSystem(
  const std::vector<T_time>& time,
  const std::vector<T_amt>& amt,
  const std::vector<T_rate>& rate,
  const std::vector<T_ii>& ii,
  const std::vector<int>& evid,
  const std::vector<int>& cmt,
  const std::vector<int>& addl,
  const std::vector<int>& ss,
  const int ncmt,
  PKParameterVectorx<T_par> pMatrix,
  PKParameterVectorx<T_biovar> biovar,
  PKParameterVectorx<T_tlag> tlag,
  const std::vector<Eigen::Matrix<T_par, Eigen::Dynamic, Eigen::Dynamic> >& linode_system) :
    events(time, amt, rate, ii, evid, cmt, addl, ss),
    parameters(time, pMatrix.v, biovar.v, tlag.v, linode_system),
    rates(),
    nKeep_(events.get_size())
  {
    events.Sort();
    parameters.Sort();
    events.AddlDoseEvents();
    parameters.CompleteParameterHistory(events);
    events.AddLagTimes(parameters, ncmt);
    rates.MakeRates(events, ncmt);
    parameters.CompleteParameterHistory(events);
  }

  // impose solvers
  // template<typename T_sol, typename T_steady_sol, template <class, class... > class T_model, class... Ts_par>
  // template<class... Ts_par, template <class... > class T_model, typename T_sol, typename T_steady_sol>
  template<typename T_sol, typename T_ssol, template <typename... > class T_model, typename... Ts_par>
  Eigen::Matrix<scalar_type, Eigen::Dynamic, Eigen::Dynamic>
  solve_with(T_sol& sol ,T_ssol& ssol) {
  // solve_with(T_sol& sol , T_steady_sol& ssol) {
    using Eigen::Matrix;
    using Eigen::Dynamic;
    using torsten::Event;
    using torsten::ModelParameters;
    using torsten::Rate;

    using model_type = T_model<Ts_par...>;

    constexpr int ncmt = model_type::ncmt;

    Matrix<scalar_type, 1, Dynamic> zeros = Matrix<scalar_type, 1, Dynamic>::Zero(ncmt);
    Matrix<scalar_type, 1, Dynamic> init = zeros;

    // COMPUTE PREDICTIONS
    Matrix<scalar_type, Dynamic, Dynamic>
      pred = Matrix<scalar_type, Dynamic, Dynamic>::Zero(nKeep_, ncmt);

  //   scalar_type Scalar = 1;  // trick to promote variables to scalar

    T_tau dt, tprev = events.get_time(0);
    Matrix<scalar_type, Dynamic, 1> pred1;
    Event<T_tau, T_amt, T_rate, T_ii> event;
    ModelParameters<T_tau, T_par, T_biovar, T_tlag> parameter;
    int iRate = 0, ikeep = 0;

    for (int i = 0; i < events.get_size(); i++) {
      event = events.GetEvent(i);

      // Use index iRate instead of i to find rate at matching time, given there
      // is one rate per time, not per event.
      if (rates.get_time(iRate) != events.get_time(i)) iRate++;
      Rate<T_tau, T_rate2> rate2;
      rate2.copy(rates.GetRate(iRate));

      for (int j = 0; j < ncmt; j++)
        rate2.get_rate_x()[j] *= parameters.GetValueBio(i, j);

      parameter = parameters.GetModelParameters(i);

      if ((event.get_evid() == 3) || (event.get_evid() == 4)) {  // reset events
        dt = 0;
        init = zeros;
      } else {
        dt = event.get_time() - tprev;
        auto model_rate = rate2.get_rate(); 
        auto model_parm = parameter.get_RealParameters(); 
        model_type pkmodel(event.get_time(), init, model_rate, model_parm);
        pred1 = sol.solve(pkmodel, dt);
        init = pred1;
      }

      if (((event.get_evid() == 1 || event.get_evid() == 4)
           && (event.get_ss() == 1 || event.get_ss() == 2)) ||
          event.get_ss() == 3) {  // steady state event
        // scalar_type Scalar = 1;  // trick to promote variables to scalar
        auto model_rate = rate2.get_rate(); 
        auto model_parm = parameter.get_RealParameters(); 
        model_type pkmodel(event.get_time(), init, model_rate, model_parm);
        pred1 = stan::math::multiply(ssol.solve(pkmodel,
                                                parameters.GetValueBio(i, event.get_cmt() - 1) * event.get_amt(),
                                                event.get_rate(),
                                                event.get_ii(),
                                                event.get_cmt()),
                                     scalar_type(1.0));

        // the object PredSS returns doesn't always have a scalar type. For
        // instance, PredSS does not depend on tlag, but pred does. So if
        // tlag were a var, the code must promote PredSS to match the type
        // of pred1. This is done by multiplying predSS by a Scalar.

        if (event.get_ss() == 2)
          init += pred1;  // steady state without reset
        else
          init = pred1;  // steady state with reset (ss = 1)
      }

      if (((event.get_evid() == 1) || (event.get_evid() == 4)) &&
          (event.get_rate() == 0)) {  // bolus dose
        init(0, event.get_cmt() - 1)
          += parameters.GetValueBio(i, event.get_cmt() - 1) * event.get_amt();
      }

      if (event.get_keep()) {
        pred.row(ikeep) = init;
        ikeep++;
      }
      tprev = event.get_time();
    }
    return pred;
  }
};

}

#endif
