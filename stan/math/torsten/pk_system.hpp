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
  template<typename T_time, typename T_init, typename T_par, typename T_rate>
  class PKOneCptModel {
    const T_time &t0_;
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0_;
    const std::vector<T_rate> &rate_;
    const T_par &CL_;
    const T_par &V2_;
    const T_par &ka_;
    const T_par k10_;
    const std::vector<T_par> alpha_;

public:
    static const int ncmt = 2; 
    using scalar_type = typename promote_args<T_time, T_rate, T_par, T_init>::type;

    // constructors
    PKOneCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const std::vector<T_rate> &rate,
                  const T_par& CL,
                  const T_par& V2,
                  const T_par& ka) :
      t0_(t0),
      y0_(y0),
      rate_(rate),
      CL_(CL),
      V2_(V2),
      ka_(ka),
      k10_(CL / V2),
      alpha_{k10_, ka_}
    {}

    PKOneCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const std::vector<T_rate> &rate,
                  const std::vector<T_par> & par) :
      PKOneCptModel(t0, y0, rate, par[0], par[1], par[2])
    {}

    PKOneCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const Eigen::Matrix<T_par, Eigen::Dynamic, Eigen::Dynamic>& m) :
      // TODO
      t0_(t0_),
      y0_(y0)
      // CL_(m.Cl_),
      // V2_(m.V2_),
      // ka_(m.ka_)
    {}

    // copy constructor
    PKOneCptModel(const PKOneCptModel& m) :
      PKOneCptModel(m.t0_, m.y0_, m.rate_, m.CL_, m.V2_, m.ka_)
    {}

    // get
    const T_time              & t0()   { return t0_; }
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0() { return y0_; }
    const std::vector<T_rate> & rate() { return rate_; }
    const T_par               & CL()   { return CL_; }
    const T_par               & V2()   { return V2_; }
    const T_par               & ka()   { return ka_; }
    const T_par               & k10()  { return k10_; }
    const std::vector<T_par>  & alpha(){ return alpha_; }

    // can be solved by linear ode solver
};

  class PKOneCptModelSolver {
  public:
    PKOneCptModelSolver() {}

    template<typename T_time, template <class, class... > class T_model, class... Ts_par>    
    Eigen::Matrix<typename T_model<Ts_par...>::scalar_type, Eigen::Dynamic, 1> 
    solve(T_model<Ts_par...> pkmodel, const T_time& dt) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using scalar_type = typename T_model<Ts_par...>::scalar_type;

      auto init = pkmodel.y0()   ;
      auto rate = pkmodel.rate() ;
      auto ka   = pkmodel.ka()   ;
      auto alpha= pkmodel.alpha();

      std::vector<scalar_type> a(2, 0);
      Matrix<scalar_type, 1, Dynamic> pred = Matrix<scalar_type, 1, Dynamic>::Zero(2);

      if ((init[0] != 0) || (rate[0] != 0)) {
        pred(0, 0) = init[0] * exp(-ka * dt) + rate[0] * (1 - exp(-ka * dt)) / ka;
        a[0] = ka / (ka - alpha[0]);
        a[1] = -a[0];
        pred(0, 1) += torsten::PolyExp(dt, init[0], 0, 0, 0, false, a, alpha, 2) +
          torsten::PolyExp(dt, 0, rate[0], dt, 0, false, a, alpha, 2);
      }

      if ((init[1] != 0) || (rate[1] != 0)) {
        a[0] = 1;
        pred(0, 1) += torsten::PolyExp(dt, init[1], 0, 0, 0, false, a, alpha, 1) +
          torsten::PolyExp(dt, 0, rate[1], dt, 0, false, a, alpha, 1);
      }
      return pred;
    }
  };

// depend on model, we can have arbitrary number of
// parameters, e.g. biovar, k12, k10, ka. Each parameter
// can be data or var.
  template<typename T_time, typename T_init, typename T_par, typename T_rate>
  class PKTwoCptModel {
    const T_time &t0_;
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0_;
    const std::vector<T_rate> &rate_;
    const T_par &CL_;
    const T_par &Q_;
    const T_par &V2_;
    const T_par &V3_;
    const T_par &ka_;
    const T_par k10_;
    const T_par k12_;
    const T_par k21_;
    const T_par ksum_;
    const std::vector<T_par> alpha_;

  public:
    static const int ncmt = 3;
    using scalar_type = typename promote_args<T_time, T_rate, T_par, T_init>::type;

    // constructors
    PKTwoCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const std::vector<T_rate> &rate,
                  const T_par& CL,
                  const T_par& Q,
                  const T_par& V2,
                  const T_par& V3,
                  const T_par& ka) :
      t0_(t0),
      y0_(y0),
      rate_(rate),
      CL_(CL),
      Q_(Q),
      V2_(V2),
      V3_(V3),
      ka_(ka),
      k10_(CL_ / V2_),
      k12_(Q_ / V2_),
      k21_(Q_ / V3_),
      ksum_(k10_ + k12_ + k21_),
      alpha_{0.5 * (ksum_ + sqrt(ksum_ * ksum_ - 4 * k10_ * k21_)),
        0.5 * (ksum_ - sqrt(ksum_ * ksum_ - 4 * k10_ * k21_)),
        ka_}
    {}

    PKTwoCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const std::vector<T_rate> &rate,
                  const std::vector<T_par> & par) :
      PKTwoCptModel(t0, y0, rate, par[0], par[1], par[2], par[3], par[4])
    {}

    PKTwoCptModel(const T_time& t0, const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0,
                  const Eigen::Matrix<T_par, Eigen::Dynamic, Eigen::Dynamic>& m) :
      // TODO
      t0_(t0_),
      y0_(y0)
      // CL_(m.Cl_),
      // V2_(m.V2_),
      // ka_(m.ka_)
    {}

    // copy constructor
    PKTwoCptModel(const PKTwoCptModel& m) :
      PKTwoCptModel(m.t0_, m.y0_, m.rate_, m.CL_, m.Q_, m.V2_, m.V3_, m.ka_)
    {}

    // get
    const T_time              & t0()   { return t0_; }
    const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& y0() { return y0_; }
    const std::vector<T_rate> & rate()  { return rate_; }
    const T_par               & CL()    { return CL_;    }
    const T_par               & Q()     { return Q_;     }
    const T_par               & V2()    { return V2_;    }
    const T_par               & V3()    { return V3_;    }
    const T_par               & ka()    { return ka_;    }
    const T_par               & k10()   { return k10_;   }
    const T_par               & k12()   { return k12_;   }
    const T_par               & k21()   { return k21_;   }
    const std::vector<T_par>  & alpha() { return alpha_; }

    // can be solved by linear ode solver
  };

  class PKTwoCptModelSolver {
  public:
    PKTwoCptModelSolver() {}

    template<typename T_time, template <class, class... > class T_model, class... Ts_par>    
    Eigen::Matrix<typename T_model<Ts_par...>::scalar_type, Eigen::Dynamic, 1> 
    solve(T_model<Ts_par...> pkmodel, const T_time& dt) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;
      using scalar_type = typename T_model<Ts_par...>::scalar_type;

      auto init = pkmodel.y0()   ;
      auto rate = pkmodel.rate() ;
      auto ka   = pkmodel.ka()   ;
      auto k10  = pkmodel.k10()  ;
      auto k12  = pkmodel.k12()  ;
      auto k21  = pkmodel.k21()  ;
      auto alpha= pkmodel.alpha();

      std::vector<scalar_type> a(3, 0);
      Matrix<scalar_type, 1, Dynamic> pred = Matrix<scalar_type, 1, Dynamic>::Zero(3);

      if ((init[0] != 0) || (rate[0] != 0))  {
        pred(0, 0) = init[0] * exp(-ka * dt) + rate[0] * (1 - exp(-ka * dt)) / ka;
        a[0] = ka * (k21 - alpha[0]) / ((ka - alpha[0]) * (alpha[1] - alpha[0]));
        a[1] = ka * (k21 - alpha[1]) / ((ka - alpha[1]) * (alpha[0] - alpha[1]));
        a[2] = -(a[0] + a[1]);
        pred(0, 1) += torsten::PolyExp(dt, init[0], 0, 0, 0, false, a, alpha, 3)
          + torsten::PolyExp(dt, 0, rate[0], dt, 0, false, a, alpha, 3);
        a[0] = ka * k12 / ((ka - alpha[0]) * (alpha[1] - alpha[0]));
        a[1] = ka * k12 / ((ka - alpha[1]) * (alpha[0] - alpha[1]));
        a[2] = -(a[0] + a[1]);
        pred(0, 2) += torsten::PolyExp(dt, init[0], 0, 0, 0, false, a, alpha, 3)
          + torsten::PolyExp(dt, 0, rate[0], dt, 0, false, a, alpha, 3);
      }

      if ((init[1] != 0) || (rate[1] != 0)) {
        a[0] = (k21 - alpha[0]) / (alpha[1] - alpha[0]);
        a[1] = (k21 - alpha[1]) / (alpha[0] - alpha[1]);
        pred(0, 1) += torsten::PolyExp(dt, init[1], 0, 0, 0, false, a, alpha, 2)
          + torsten::PolyExp(dt, 0, rate[1], dt, 0, false, a, alpha, 2);
        a[0] = k12 / (alpha[1] - alpha[0]);
        a[1] = -a[0];
        pred(0, 2) += torsten::PolyExp(dt, init[1], 0, 0, 0, false, a, alpha, 2)
          + torsten::PolyExp(dt, 0, rate[1], dt, 0, false, a, alpha, 2);
      }

      if ((init[2] != 0) || (rate[2] != 0)) {
        a[0] = k21 / (alpha[1] - alpha[0]);
        a[1] = -a[0];
        pred(0, 1) += torsten::PolyExp(dt, init[2], 0, 0, 0, false, a, alpha, 2)
          + torsten::PolyExp(dt, 0, rate[2], dt, 0, false, a, alpha, 2);
        a[0] = (k10 + k12 - alpha[0]) / (alpha[1] - alpha[0]);
        a[1] = (k10 + k12 - alpha[1]) / (alpha[0] - alpha[1]);
        pred(0, 2) += torsten::PolyExp(dt, init[2], 0, 0, 0, false, a, alpha, 2)
          + torsten::PolyExp(dt, 0, rate[2], dt, 0, false, a, alpha, 2);
      }

      return pred;
    }
  };

  class PKOneCptModelSolverSS {
  public:
    PKOneCptModelSolverSS () {}

    template<typename T_amt, typename T_rate, 
             typename T_ii,
             template <class, class... > class T_model, class... Ts_par>
    Eigen::Matrix<typename T_model<Ts_par...>::scalar_type, Eigen::Dynamic, 1> 
    solve(T_model<Ts_par...> pkmodel,
          const T_amt& amt,
          const T_rate& rate,
          const T_ii& ii,
          const int& cmt) const {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;
      using scalar_type = typename T_model<Ts_par...>::scalar_type;

      const double inf = std::numeric_limits<double>::max();  // "infinity"

      auto ka   = pkmodel.ka()   ;
      auto alpha= pkmodel.alpha();

      std::vector<scalar_type> a(2, 0);
      Matrix<scalar_type, 1, Dynamic> pred = Matrix<scalar_type, 1, Dynamic>::Zero(2);
      if (rate == 0) {  // bolus dose
        if (cmt == 1) {
          a[0] = 0;
          a[1] = 1;
          pred(0) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 2);
          a[0] = ka / (ka - alpha[0]);
          a[1] = -a[0];
          pred(1) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 2);
        } else {  // cmt=2
          a[0] = 1;
          pred(1) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 1);
        }
      } else if (ii > 0) {  // multiple truncated infusions
        double delta = torsten::unpromote(amt / rate);
        static const char* function("Steady State Event");
        torsten::check_mti(amt, delta, ii, function);

        if (cmt == 1) {
          a[0] = 0;
          a[1] = 1;
          pred(0) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 2);
          a[0] = ka / (ka - alpha[0]);
          a[1] = -a[0];
          pred(1) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 2);
        } else {  // cmt = 2
          a[0] = 1;
          pred(1) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 1);
        }
      } else {  // constant infusion
        if (cmt == 1) {
          a[0] = 0;
          a[1] = 1;
          pred(0) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 2);
          a[0] = ka / (ka - alpha[0]);
          a[1] = -a[0];
          pred(1) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 2);
        } else {  // cmt = 2
          a[0] = 1;
          pred(1) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 1);
        }
      }
      return pred;
    }
  };

  class PKTwoCptModelSolverSS {
  public:
    PKTwoCptModelSolverSS() {}

    template<typename T_amt, typename T_rate, 
             typename T_ii,
             template <class, class... > class T_model, class... Ts_par>
    Eigen::Matrix<typename T_model<Ts_par...>::scalar_type, Eigen::Dynamic, 1> 
    solve(T_model<Ts_par...> pkmodel,
          const T_amt& amt,
          const T_rate& rate,
          const T_ii& ii,
          const int& cmt) const {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;
      using scalar_type = typename T_model<Ts_par...>::scalar_type;

      const double inf = std::numeric_limits<double>::max();  // "infinity"

      auto init = pkmodel.y0()   ;
      auto ka   = pkmodel.ka()   ;
      auto k10  = pkmodel.k10()  ;
      auto k12  = pkmodel.k12()  ;
      auto k21  = pkmodel.k21()  ;
      auto alpha= pkmodel.alpha();

      std::vector<scalar_type> a(3, 0);
      Matrix<scalar_type, 1, Dynamic> pred = Matrix<scalar_type, 1, Dynamic>::Zero(3);

      if (rate == 0) {  // bolus dose
        if (cmt == 1) {
          pred(0, 0) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 3);
          a[0] = ka * (k21 - alpha[0]) / ((ka - alpha[0])
                                          * (alpha[1] - alpha[0]));
          a[1] = ka * (k21 - alpha[1]) / ((ka - alpha[1])
                                          * (alpha[0] - alpha[1]));
          a[2] = -(a[0] + a[1]);
          pred(0, 1) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 3);
          a[0] = ka * k12 / ((ka - alpha[0]) * (alpha[1] - alpha[0]));
          a[1] = ka * k12 / ((ka - alpha[1]) * (alpha[0] - alpha[1]));
          a[2] = -(a[0] + a[1]);
          pred(0, 2) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 3);
        } else if (cmt == 2) {
          a[0] = (k21 - alpha[0]) / (alpha[1] - alpha[0]);
          a[1] = (k21 - alpha[1]) / (alpha[0] - alpha[1]);
          pred(0, 1) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 2);
          a[0] = ka * k12 / ((ka - alpha[0]) * (alpha[1] - alpha[0]));
          a[1] = ka * k12 / ((ka - alpha[1]) * (alpha[0] - alpha[1]));
          pred(0, 2) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 3);
        } else {  // cmt=3
          a[0] = k21 / (alpha[1] - alpha[0]);
          a[1] = -a[0];
          pred(0, 1) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 2);
          a[0] = (k10 + k12 - alpha[0]) / (alpha[1] - alpha[0]);
          a[1] = (k10 + k12 - alpha[1]) / (alpha[0] - alpha[1]);
          pred(0, 2) = torsten::PolyExp(ii, amt, 0, 0, ii, true, a, alpha, 2);
        }
      } else if (ii > 0) {  // multiple truncated infusions
        double delta = torsten::unpromote(amt / rate);
        static const char* function("Steady State Event");
        torsten::check_mti(amt, delta, ii, function);

        if (cmt == 1) {
          a[2] = 1;
          pred(0, 0) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 3);
          a[0] = ka * (k21 - alpha[0]) / ((ka - alpha[0])
                                          * (alpha[1] - alpha[0]));
          a[1] = ka * (k21 - alpha[1]) / ((ka - alpha[1])
                                          * (alpha[0] - alpha[1]));
          a[2] = - (a[0] + a[1]);
          pred(0, 1) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 3);
          a[0] = ka * k12 / ((ka - alpha[0]) * (alpha[1] - alpha[0]));
          a[1] = ka * k12 / ((ka - alpha[1]) * (alpha[0] - alpha[1]));
          a[2] = -(a[0] + a[1]);
          pred(0, 2) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 3);
        } else if (cmt == 2) {
          a[0] = (k21 - alpha[0]) / (alpha[1] - alpha[0]);
          a[1] = (k21 - alpha[1]) / (alpha[0] - alpha[1]);
          pred(0, 1) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 2);
          a[0] = k12 / (alpha[1] - alpha[0]);
          a[1] = -a[0];
          pred(0, 2) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 2);
        } else {  // cmt=3
          a[0] = k21 / (alpha[1] - alpha[0]);
          a[1] = -a[0];
          pred(0, 1) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 2);
          a[0] = (k10 + k12 - alpha[0]) / (alpha[1] - alpha[0]);
          a[1] = (k10 + k12 - alpha[1]) / (alpha[0] - alpha[1]);
          pred(0, 2) = torsten::PolyExp(ii, 0, rate, amt / rate, ii, true, a, alpha, 2);
        }
      } else {  // constant infusion
        if (cmt == 1) {
          a[2] = 1;
          pred(0, 0) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 3);
          a[0] = ka * (k21 - alpha[0]) / ((ka - alpha[0])
                                          * (alpha[1] - alpha[0]));
          a[1] = ka * (k21 - alpha[1]) / ((ka - alpha[1])
                                          * (alpha[0] - alpha[1]));
          a[2] = -(a[0] + a[1]);
          pred(0, 1) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 3);
          a[0] = ka * k12 / ((ka - alpha[0]) * (alpha[1] - alpha[0]));
          a[1] = ka * k12 / ((ka - alpha[1]) * (alpha[0] - alpha[1]));
          a[2] = -(a[0] + a[1]);
          pred(0, 2) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 3);
        } else if (cmt == 2) {
          a[0] = (k21 - alpha[0]) / (alpha[1] - alpha[0]);
          a[1] = (k21 - alpha[1]) / (alpha[0] - alpha[1]);
          pred(0, 1) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 2);
          a[0] = k12 / (alpha[1] - alpha[0]);
          a[1] = -a[0];
          pred(0, 2) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 2);
        } else {  // cmt=3
          a[0] = k21 / (alpha[1] - alpha[0]);
          a[1] = -a[0];
          pred(0, 1) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 2);
          a[0] = (k10 + k12 - alpha[0]) / (alpha[1] - alpha[0]);
          a[1] = (k10 + k12 - alpha[1]) / (alpha[0] - alpha[1]);
          pred(0, 2) = torsten::PolyExp(0, 0, rate, inf, 0, true, a, alpha, 2);
        }
      }
      return pred;
    }
  };

  // conversion constructor and indexing sugar
  template<typename T>
  struct PKParameterVector {
    std::vector<std::vector<T> > v;

    PKParameterVector(std::vector<std::vector<T> > v0) : v(v0) {}

    PKParameterVector(std::vector<T> v0) : v({1, v0}) {}

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
  PKParameterVector<T_par> pMatrix,
  PKParameterVector<T_biovar> biovar,
  PKParameterVector<T_tlag> tlag,
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
