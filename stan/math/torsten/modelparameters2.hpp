#ifndef MODELPARAMETERS2_HPP
#define MODELPARAMETERS2_HPP

#include <Eigen/Dense>
// #include <stan/math/torsten/PKModel/Event.hpp>
#include <stan/math/torsten/PKModel/ExtractVector.hpp>
#include <stan/math/torsten/PKModel/SearchReal.hpp>
#include <algorithm>
#include <vector>

namespace refactor {

// template<typename T_time,
//          typename T_parameters,
//          typename T_biovar,
//          typename T_tlag>
//   class ModelParameterHistory;
/**
 * The ModelParameters class defines objects that contain the parameters of
 * a model at a given time.
 */
template<typename T_time,
         typename T_parameters,
         typename T_biovar,
         typename T_tlag>
struct ModelParameters {
  T_time time;
  std::vector<T_parameters> theta;
  std::vector<T_biovar> biovar;
  std::vector<T_tlag> tlag;
  Eigen::Matrix<T_parameters, Eigen::Dynamic, Eigen::Dynamic> K;

  // constructors
  ModelParameters() :
    time{0.0}, theta{0.0}, biovar{0.0}, tlag{0.0}
  {}

  ModelParameters(const T_time& time_in,
                  const std::vector<T_parameters>& theta_in,
                  const std::vector<T_biovar>& biovar_in,
                  const std::vector<T_tlag>& tlag_in,
                  const Eigen::Matrix<T_parameters, Eigen::Dynamic, Eigen::Dynamic>& K_in)
    : time(time_in), theta(theta_in), biovar(biovar_in), tlag(tlag_in), K(K_in)
  {}

  ModelParameters(const T_time& time_in,
                  const std::vector<T_parameters>& theta_in,
                  const std::vector<T_biovar>& biovar_in,
                  const std::vector<T_tlag>& tlag_in)
    : time(time_in), theta(theta_in), biovar(biovar_in), tlag(tlag_in)
  {}

  // copy constructor
  ModelParameters(const ModelParameters& par)
    : time(par.time), theta(par.theta), biovar(par.biovar), tlag(par.tlag), K(par.K)  
  {}

  /**
   * Returns a model parameter object which only contain
   * the n first parameters. This is useful for the
   * mixed solver: when we compute the base analytical
   * solution, we only want to pass the PK parameters
   * (as oppose to all the PK/PD parameters).
   */
  // ModelParameters<T_time, T_parameters, T_biovar, T_tlag>
  // truncate(int n) const {
  //   std::vector<T_parameters> tr_theta(n);
  //   for (int i = 0; i < n; i++) tr_theta[i] = theta_[i];
  //   return ModelParameters(time_, tr_theta, biovar_, tlag_, K_);
  // }

  /**
   * Adds parameters. Useful for the mixed solver, where
   * we want to augment the parameters with the intial PK
   * states when calling the numerical integrator.
   */
  // template <typename T>
  // ModelParameters<T_time, T, T_biovar, T_tlag>
  // augment(const std::vector<T>& thetaAdd) const {
  //   std::vector<T> theta(theta_.size());
  //   for (size_t i = 0; i < theta.size(); i++) theta[i] = theta_[i];
  //   for (size_t i = 0; i < thetaAdd.size(); i++) theta.push_back(thetaAdd[i]);
  //   return
  //     ModelParameters<T_time, T, T_biovar, T_tlag>
  //       (time_, theta, biovar_, tlag_);
  // }

  // template <typename T>
  // ModelParameters<T_time, T, T_biovar, T_tlag>
  // augment(const Eigen::Matrix<T, Eigen::Dynamic, 1>& thetaAdd)
  // const {
  //   return augment(stan::math::to_array_1d(thetaAdd));
  // }

  /**
   * Edit time stored in parameter object.
   */
  // void time(double time) {
  //   time_ = time;
  // }

  // int CountParameters() const {
  //   return theta_.size();
  // }

    friend std::ostream& operator<<(std::ostream& os, 
                                    const ModelParameters<T_time, T_parameters, T_biovar, T_tlag>& par) {
      os << "time: " << "[";
      for (auto ii : par.time) os << " " << *ii;
      os << " ]" << "\n";

      os << "theta: " << "[";
      for (auto ii : par.theta) os << " " << *ii;
      os << " ]" << "\n";

      os << "biovar: " << "[";
      for (auto ii : par.biovar) os << " " << *ii;
      os << " ]" << "\n";

      os << "tlag: " << "[";
      for (auto ii : par.tlag) os << " " << *ii;
      os << " ]" << "\n";

      os << par.K;

      return os;
    }

};

/**
 * The ModelParameterHistory class defines objects that contain a vector
 * of ModelParameters, along with a series of functions that operate on
 * them.
 */
template<typename T_time,
         typename T_parameters,
         typename T_biovar,
         typename T_tlag>
struct ModelParameterHistory {
  std::vector< ModelParameters<T_time, T_parameters,
                               T_biovar, T_tlag> > mpv;

  // construcotrs
  template<typename T0, typename T1, typename T2, typename T3>
  ModelParameterHistory(std::vector<T0> time,
                        std::vector<std::vector<T1> > theta,
                        std::vector<std::vector<T2> > biovar,
                        std::vector<std::vector<T3> > tlag,
                        std::vector< Eigen::Matrix<T1, Eigen::Dynamic, Eigen::Dynamic> > K) 
  {
    int n = std::max( {theta.size(), K.size(), biovar.size(), tlag.size() } );
    mpv.reserve(n);
    int j, k, l, m;
    // FIX ME - is this the most efficient way of storing data?
    for (int i = 0; i < n; i++) {
      (theta.size() == 1) ? j = 0 : j = i;
      (biovar.size() == 1) ? k = 0 : k = i;
      (tlag.size() == 1) ? l = 0 : l = i;
      (K.size() == 1) ? m = 0 : m = i;
      mpv.push_back(ModelParameters<T_time, T_parameters, T_biovar, T_tlag>
                    (time[i], theta[j], biovar[k], tlag[l], K[m]) );
    }
  }

  // indexing
  ModelParameters<T_time, T_parameters, T_biovar, T_tlag>& operator[](const int index) {
    return mpv[index];
  }
 
  const ModelParameters<T_time, T_parameters, T_biovar, T_tlag>& operator[](const int index) const {
    return mpv[index];
  }

  /**
   * COMPLETE MODEL PARAMETERS
   *
   * Completes parameters so that it contains model parameters for each event 
   * in events. If parameters contains only one set of parameters (case where
   * the parameters are constant), this set is replicated for each event in
   * events. Otherwise a new parameter vector is added for each new event 
   * (isnew = true). This parameter vector is identical to the parameter vector
   * at the subsequent event. If the new event occurs at a time posterior to
   * the time of the last event, than the new vector parameter equals the
   * parameter vector of the last event. This amounts to doing an LOCF
   * (Last Observation Carried Forward).
   *
   * Events and Parameters are sorted at the end of the procedure.
   *
   * @param[in] parameters at each event
   * @param[in] events elements (following NONMEM convention) at each event
   * @return - modified parameters and events.
   */
  template<typename T0, typename T1, typename T2, typename T3>
  void CompleteParameterHistory(refactor::PKEventList<T0, T1, T2, T3> & events) {
    int nEvent = events.Events.size();
    assert(nEvent > 0);
    int len_Parameters = mpv.size();  // numbers of events for which parameters
                                       // are determined
    assert(len_Parameters > 0);

    sort_by_time(mpv);
    sort_by_time(events.Events);
    mpv.resize(nEvent);

    int iEvent = 0;
    for (int i = 0; i < len_Parameters - 1; i++) {
      while (events.Events[i].isnew) iEvent++;  // skip new events
      assert(mpv[i].time_ == events.Events[iEvent].time);  // compare time of
                                                         // "old' events to
                                                         // time of
                                                         // parameters.
      iEvent++;
    }

    // if (len_Parameters == 1)  {
    //   for (int i = 0; i < nEvent; i++) {
    //     // FIX ME - inefficient data storage
    //     mpv[i].theta_ = mpv[0].theta_;
    //     mpv[i].biovar_ = mpv[0].biovar_;
    //     mpv[i].tlag_ = mpv[0].tlag_;
    //     mpv[i].K_ = mpv[0].K_;
    //     mpv[i].time_ = events.Events[i].time;
    //     events.Events[i].isnew = false;
    //   }
    // } else {  // parameters are event dependent.
    //   std::vector<T_time> times(nEvent, 0);
    //   for (int i = 0; i < nEvent; i++) times[i] = mpv[i].time_;
    //   iEvent = 0;

    //   int k, j = 0;
    //   ModelParameters<T_time, T_parameters, T_biovar, T_tlag> newParameter;

    //   for (int i = 0; i < nEvent; i++) {
    //     while (events.Events[iEvent].isnew) {
    //       /* Three cases:
    //        * (a) The time of the new event is higher than the time of the last
    //        *     parameter vector in parameters (k = len_parameters).
    //        *     Create a parameter vector at the the time of the new event,
    //        *     with the parameters of the last parameter vector.
    //        *     (Last Observation Carried Forward)
    //        * (b) The time of the new event matches the time of a parameter vector
    //        *     in parameters. This parameter vector gets replicated.
    //        * (c) (a) is not verified and no parameter vector occurs at the time
    //        *     of the new event. A new parameter vector is created at the time
    //        *     of the new event, and its parameters are equal to the parameters
    //        *     of the subsequent parameter vector in parameters.
    //        */
    //       // Find the index corresponding to the time of the new event in the
    //       // times vector.
    //       k = torsten::SearchReal(times, len_Parameters - 1, events.Events[iEvent].time);

    //       if ((k == len_Parameters) ||
    //         (events.Events[iEvent].time == mpv[k - 1].time_))
    //         newParameter = GetModelParameters(k - 1);
    //       else
    //         newParameter = GetModelParameters(k);

    //       newParameter.time_ = events.Events[iEvent].time;
    //       mpv[len_Parameters + j] = newParameter;
    //       events.Events[iEvent].isnew = false;
    //       if (iEvent < nEvent - 1) iEvent++;
    //       j++;
    //     }

    //     if (iEvent < nEvent - 1) iEvent++;
    //   }
    // }
    sort_by_time(mpv);
  }
  
//   // declare friends
//   friend class ModelParameters<T_time, T_parameters, T_biovar, T_tlag>;
//   template<typename T1, typename T2, typename T3, typename T4>
//     friend class Events;
};

} 

#endif
