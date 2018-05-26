#ifndef PK_ODE_SOLVER_HPP
#define PK_ODE_SOLVER_HPP

#include <stan/math/torsten/PKModel/functors/general_functor.hpp>
#include <stan/math/torsten/pk_rate_adaptor.hpp>
#include <stan/math/torsten/pk_adapt_ode_solver.hpp>

namespace refactor {

  using boost::math::tools::promote_args;
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::to_array_1d;
  using std::vector;
  using boost::math::tools::promote_args;


  class PKODEModelSolver {
    torsten::integrator_structure integrator_;
    
    template<typename T_scalar, typename F, typename T_par, typename T_rate, typename T_init>
    void run_integrator(const F& f,
                   const std::vector<T_par>& parameters,
                   const std::vector<T_rate>& rate,
                   const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& init,
                   double& InitTime_d,
                   std::vector<double>& EventTime_d,
                   Eigen::Matrix<T_scalar, Eigen::Dynamic, 1>& pred) const
    {
      const std::vector<T_par>& theta = parameters;

        if (EventTime_d[0] == InitTime_d) {
          pred = init;
        } else {
          vector<T_scalar> init_vector = to_array_1d(init);
          vector<double> x_r;

          vector<int> idummy;
          vector<vector<T_scalar> >
            pred_V = integrator_(f,
                                 init_vector,
                                 InitTime_d,
                                 EventTime_d,
                                 theta,
                                 x_r,
                                 idummy);

          // Convert vector in row-major vector (eigen Matrix)
          // FIX ME - want to return column-major vector to use Stan's
          // to_vector function.
          // std::cout << "taki test: " << pred_V[0].size() << "\n";
          pred = stan::math::to_vector(pred_V[0]);
          // pred.resize(pred_V[0].size());
          // for (size_t i = 0; i < pred_V[0].size(); i++) pred(0, i) = pred_V[0][i];
        }
    }

    // overload for data rate
    template<typename T_scalar, typename F, typename T_par, typename T_init>
    void
    run_integrator(const F& f,
                   const std::vector<T_par>& parameters,
                   const std::vector<double>& rate,
                   const Eigen::Matrix<T_init, 1, Eigen::Dynamic>& init,
                   double& InitTime_d,
                   std::vector<double>& EventTime_d,
                   Eigen::Matrix<T_scalar, Eigen::Dynamic, 1>& pred) const
    {
      const std::vector<T_par>& theta = parameters;

        if (EventTime_d[0] == InitTime_d) { pred = init;
        } else {
        vector<T_scalar> init_vector = to_array_1d(init);
          vector<int> idummy;
          vector<vector<T_scalar> >
            pred_V = integrator_(f,
                                 init_vector,
                                 InitTime_d,
                                 EventTime_d,
                                 theta,
                                 rate,
                                 idummy);

          // std::cout << "taki test: " << pred_V[0].size() << "\n";
          // Convert vector in row-major vector (eigen Matrix)
          pred = stan::math::to_vector(pred_V[0]);
          // pred.resize(pred_V[0].size());
          // for (size_t i = 0; i < pred_V[0].size(); i++) pred(i) = pred_V[0][i];
        }
    }

  public:
    // constructor
    PKODEModelSolver(const double& rel_tol,
                     const double& abs_tol,
                     const long int& max_num_steps,
                     std::ostream* msgs,
                     const std::string& integratorType) :
      integrator_(rel_tol, abs_tol, max_num_steps, msgs, integratorType)
    {}

    // constructor
    PKODEModelSolver(const torsten::integrator_structure integrator) :
      integrator_(integrator)
    {}

    const torsten::integrator_structure& integrator() const {
      return integrator_;
    }


    template<typename T_time, template <class, class... > class T_model, class... Ts_par>    
    Eigen::Matrix<typename T_model<Ts_par...>::scalar_type, Eigen::Dynamic, 1> 
    solve(const T_model<Ts_par...> &pkmodel, const T_time& dt) const {
      PKAdaptODESolver<PKODERateAdaptor<T_model<Ts_par...>>>
                                adapted_sol(integrator_);
      return adapted_sol.solve(pkmodel,dt);
    }

    // WIP, for SS solver
    template<typename F>
    Eigen::Matrix<double, Eigen::Dynamic, 1> 
    solve(const F& f,
          const double &t0,
          const Eigen::Matrix<double, 1, Eigen::Dynamic>& init,
          const double& dt,
          const std::vector<double>& pars,
          const std::vector<double>& rate){
      using stan::math::to_array_1d;
      using std::vector;

      std::vector<double> EventTime {t0};

      assert((size_t) init.cols() == rate.size());

      double InitTime = t0 - dt;  // time of previous event

      Eigen::Matrix<double, Eigen::Dynamic, 1> pred;
      
      run_integrator(f,
                     pars,
                     rate,
                     init,
                     InitTime,
                     EventTime,
                     pred);
      return pred;
    }
  };
}




#endif
