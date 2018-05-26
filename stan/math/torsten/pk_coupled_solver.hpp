#ifndef PK_COUPLED_SOLVER_HPP
#define PK_COUPLED_SOLVER_HPP

#include <stan/math/torsten/PKModel/functors/general_functor.hpp>
#include <stan/math/torsten/pk_cptode_adaptor.hpp>
#include <stan/math/torsten/pk_onecpt_solver.hpp>
#include <stan/math/torsten/pk_twocpt_solver.hpp>

namespace refactor {

  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::math::to_array_1d;
  using std::vector;
  using boost::math::tools::promote_args;

  /**
   * Coupled model solvers are composed of two solvers accordingly.
   *
   * @tparam T_s1 type of solver 1
   * @tparam T_s2 type of solver 2
   */
  template<typename T_s1, typename T_s2>
  class PKCoupledModelSolver {
    const T_s1 &sol1_;
    const T_s2 &sol2_;
    
  public:
    /**
     * Coupled model solvers are composed of two solvers.
     *
     * @tparam sol1 solver 1
     * @tparam sol2 solver 2
     */
    PKCoupledModelSolver(const T_s1 &sol1, const T_s2 &sol2) :
      sol1_(sol1), sol2_(sol2)
    {}

  /**
   * Solving a coupled model. This is largely formal because
   * most of the time we want some interactions between the
   * models, and the solution process should reflect that
   * through adaptors.
   *
   * @tparam T_time type of dt
   * @tparam T_model1 type of model 1
   * @tparam T_model2 type of model 2
   * @tparam Ts1 type of model 1's parameters
   * @tparam Ts2 type of model 2's parameters
   */
    template<typename T_time,
             template <typename...> class T_model1,
             template <typename...> class T_model2,
             typename... Ts1,
             typename... Ts2>
    Eigen::Matrix<typename
                  PKCoupledModel2<T_model1<Ts1...>,
                                  T_model2<Ts2...> >::scalar_type, 
                  1, Eigen::Dynamic>
    solve(const PKCoupledModel2
          <T_model1<Ts1...>, T_model2<Ts2...> > &coupled_model, 
          const T_time& dt) const {
      using model_type = PKCoupledModel2<T_model1<Ts1...>, T_model2<Ts2...> >;
      using scalar = typename model_type::scalar_type;

      Eigen::Matrix<scalar, 1, Eigen::Dynamic> pred;

      const T_model1<Ts1...> & model1 = coupled_model.model1();
      const T_model2<Ts2...> & model2 = coupled_model.model2();
      auto x1 = sol1_.solve(model1, dt);
      auto x2 = sol2_.solve(model2, dt);

      pred.resize(x1.size() + x2.size());
      pred << x1.transpose(), x2.transpose();

      return pred;
    }
    
  /**
   * Solving a PK comparment-ODE coupled model. The
   * PKCptODEAdaptor deals with interaction and data
   * exchange between model1(pk model) and model2(ODE model).
   * This is used for combining analytical solution of PK
   * models with numerical integration of ODEs, in order to
   * increase efficiency.
   *
   * @tparam T0 type of dt
   * @tparam T_model type of coupled model.
   * @tparam Ts type of model parameters
   * @param model coupled model.
   * @param dt time span
   */
    template<typename T0,
             template<typename...> class T_model,
             typename... Ts>
    Matrix<typename T_model<Ts...>::scalar_type, Dynamic, 1>
    solve(const T_model<Ts...> &model, const T0& dt) const {
      torsten::integrator_structure integrator(sol2_.integrator());
      using Adaptor = PKCptODEAdaptor<T_model<Ts...>, T_s1>;
      PKAdaptODESolver<Adaptor> adapted_sol(integrator);
      auto model1 = model.model.model1();
      auto model2 = model.model.model2();

      using scalar = typename T_model<Ts...>::scalar_type;
      Eigen::Matrix<scalar, 1, Eigen::Dynamic> pred;
      if (stan::math::value_of(dt) < 1.e-9) {
        pred.resize(model1.y0().size() + model2.y0().size());
        pred << model1.y0(), model2.y0();
      } else {
        auto x1 = sol1_.solve(model1, dt);
        auto x2 = adapted_sol.solve(model, dt);
        pred.resize(x1.size() + x2.size());
        pred << x1.transpose(), x2.transpose();
      }
      return pred;      
    }
  };
}
#endif
