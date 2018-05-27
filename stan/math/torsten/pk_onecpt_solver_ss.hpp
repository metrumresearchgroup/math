#ifndef PK_ONECPT_SOLVER_SS_HPP
#define PK_ONECPT_SOLVER_SS_HPP

namespace refactor {

  using boost::math::tools::promote_args;

  /**
   * Standard one compartment steady state PK ODE solver based on
   * analytical solution and matrix exponential.
   */
  class PKOneCptModelSolverSS {
  public:
    /**
     * Constructor
     */
    PKOneCptModelSolverSS () {}

  /**
   * standard one compartment PK model attached to this solver.
   * @tparam T_time t type
   * @tparam T_init initial condition type
   * @tparam T_rate rate type
   * @tparam T_par parameter type
   */
    template<typename T_time, typename T_init, typename T_rate, typename T_par>
    using default_model = PKOneCptModel<T_time, T_init, T_rate, T_par>;

  /**
   * Solve one-cpt steady state model.
   *
   * @tparam T_amt amt type
   * @tparam T_rate dosing rate type
   * @tparam T_ii dosing interval type.
   * @tparam T_model model type
   * @tparam Ts_par model parameter type
   * @param pkmodel PK one-cpt model
   * @param amt dosing amount
   * @param rate dosing rate
   * @param ii dosing interval
   * @param cmt dosing compartment
   */
    template<typename T_amt, typename T_rate, 
             typename T_ii,
             template <class, class... > class T_model,
             class... Ts_par>
    PKRecord<typename promote_args<
               T_amt, T_rate, T_ii,
               typename T_model<Ts_par...>::par_type>::type>
    solve(const T_model<Ts_par...> &pkmodel,
          const T_amt& amt,
          const T_rate& rate,
          const T_ii& ii,
          const int& cmt) const {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using std::vector;
      using scalar_type = typename promote_args<
        T_amt, T_rate, T_ii, typename T_model<Ts_par...>::par_type>::type;

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

}

#endif
