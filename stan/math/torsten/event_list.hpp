#ifndef EVENTHISTORY_HPP
#define EVENTHISTORY_HPP

#include <iostream>     // std::cout, std::right, std::endl
#include <iomanip>      // std::setw
// #include <stan/math/torsten/modelparameters2.hpp>

namespace refactor {

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

  template <typename T_time_event, typename T_amt, typename T_rate, typename T_ii,
            typename T_par, typename T_biovar, typename T_tlag>
  struct PKEvent {
    T_time_event time;
    T_amt amt;
    T_rate rate;
    T_ii ii;
    int evid, cmt, addl, ss;
    bool keep, isnew;
    std::vector<T_par> theta;
    std::vector<T_biovar> biovar;
    std::vector<T_tlag> tlag;

    PKEvent() :
      time (0),
      amt  (0),
      rate (0),
      ii   (0),
      evid (0),
      cmt  (0),
      addl (0),
      ss   (0),
      keep (false),
      isnew(false)
    {}

    PKEvent(T_time_event p_time,
            T_amt p_amt,
            T_rate p_rate,
            T_ii p_ii,
            int p_evid,
            int p_cmt,
            int p_addl,
            int p_ss,
            bool p_keep,
            bool p_isnew,
            std::vector<T_par>& theta_in,
            std::vector<T_biovar>& biovar_in,
            std::vector<T_tlag>& tlag_in) :
      time   (p_time),
      amt    (p_amt),
      rate   (p_rate),
      ii     (p_ii),
      evid   (p_evid),
      cmt    (p_cmt),
      addl   (p_addl),
      ss     (p_ss),
      keep   (p_keep),
      isnew  (p_isnew),
      theta  (theta_in),
      biovar (biovar_in),
      tlag   (tlag_in)
    {}

    PKEvent(const PKEvent& ev) {
      time   = ev.time;
      amt    = ev.amt;
      rate   = ev.rate;
      ii     = ev.ii;
      evid   = ev.evid;
      cmt    = ev.cmt;
      addl   = ev.addl;
      ss     = ev.ss;
      keep   = ev.keep;
      isnew  = ev.isnew;
      theta  = ev.theta;
      biovar = ev.biovar;
      tlag   = ev.tlag;
    }

    inline bool has_dosing() {
      return (evid == 1 || evid == 4);
    }

    inline bool has_addl() {
      return has_dosing() && (addl > 0 && ii > 0);
    }
      
    friend std::ostream& operator<<(std::ostream& os, 
                                    const PKEvent<T_time_event, T_amt, T_rate, T_ii, T_par, T_biovar, T_tlag>& ev) {
      return os << ev.time << " "
                << ev.amt  << " "
                << ev.rate << " "
                << ev.ii   << " "
                << ev.evid << " "
                << ev.cmt  << " "
                << ev.addl << " "
                << ev.ss   << " "
                << ev.keep << " "
                << ev.isnew;
    }
  };

  // PK parameters can be event-specific or uniform(for all events)
  // use coersion for different types of arguments
  // template<typename T>
  // struct PKParameterVector {
  //   std::vector<std::vector<T> > v;

  //   PKParameterVector(std::vector<std::vector<T> > v0) : v(v0) {}
  //   // operator std::vector<std::vector<T> > () const { return v; }

  //   PKParameterVector(std::vector<T> v0) : v({1, v0}) {}

  //   std::vector<T>& operator[](const int index) {
  //     return v.size() == 1 ? v[0] : v[index];
  //   }

  //   const std::vector<T>& operator[](const int index) const {
  //     return v.size() == 1 ? v[0] : v[index];
  //   }
  // };

  /* used to be EventHistory, but...this is what NONMEN provides, so
   * there is no 1-1 mapping between even list and PK ODE
   * parameters, which is described by PKSystem.
   */
  template <typename T_time_event, typename T_amt, typename T_rate, typename T_ii,
            typename T_par, typename T_biovar, typename T_tlag>
  struct PKEventList {
    using List = std::vector<PKEvent<T_time_event, T_amt, T_rate, T_ii, T_par, T_biovar, T_tlag> >;

    List Events;

    // constructor
    PKEventList(std::vector<T_time_event> p_time,
                std::vector<T_amt> p_amt,
                std::vector<T_rate> p_rate,
                std::vector<T_ii> p_ii,
                std::vector<int> p_evid,
                std::vector<int> p_cmt,
                std::vector<int> p_addl,
                std::vector<int> p_ss,
                PKParameterVector<T_par> pMatrix,
                PKParameterVector<T_biovar> biovar,
                PKParameterVector<T_tlag> tlag) {
      int nEvent = p_time.size();
      if (p_ii.size() == 1) {
        p_ii.assign(nEvent, 0);
        p_addl.assign(nEvent, 0);
      }
      if (p_ss.size() == 1) p_ss.assign(nEvent, 0);

      Events.resize(nEvent);
      for (int i = 0; i < nEvent; i++) {
        Events[i] =
          PKEvent<T_time_event, T_amt, T_rate, T_ii, T_par,
                  T_biovar, T_tlag>(p_time[i],
                                    p_amt[i],
                                    p_rate[i],
                                    p_ii[i],
                                    p_evid[i],
                                    p_cmt[i],
                                    p_addl[i],
                                    p_ss[i],
                                    true,
                                    false,
                                    pMatrix[i],
                                    biovar[i],
                                    tlag[i]);
      }

      sort_by_time(Events);

      // addl dose
      for (auto ev: Events) {
        if (ev.has_addl()) {
          PKEvent<T_time_event, T_amt, T_rate, T_ii, T_par, T_biovar, T_tlag> newEvent{ev};
          newEvent.addl = 0;
          newEvent.ii = 0;
          newEvent.ss = 0;
          newEvent.keep = false;
          newEvent.isnew = true;
          for (int j = 0; j < ev.addl; j++) {
            newEvent.time = ev.time + (j+1) * ev.ii;
            Events.push_back(newEvent);
          }
        }
      }

      sort_by_time(Events);

    }
    
    // template<std::vector<
    // void sort_by_time() {
    //   struct by_time {
    //     bool operator()(const PKEvent<T_time_event, T_amt, T_rate, T_ii> &a,
    //                     const PKEvent<T_time_event, T_amt, T_rate, T_ii> &b) {
    //       return a.time < b.time;
    //     }
    //   };

    //   std::stable_sort(Events.begin(), Events.end(), by_time());
    // }


    inline bool is_sorted() {
      int i = Events.size() - 1;
      bool ordered = true;

      while ((i > 0) && (ordered)) {
        // note: evid = 3 and evid = 4 correspond to reset events
        ordered = (((Events[i].time >= Events[i - 1].time)
                    || (Events[i].evid == 3)) || (Events[i].evid == 4));
        i--;
      }
      return ordered;
    }

    // indexing
    PKEvent<T_time_event, T_amt, T_rate, T_ii, T_par, T_biovar, T_tlag>& operator[](const int index) {
      return Events[index];
    }
 
    const PKEvent<T_time_event, T_amt, T_rate, T_ii, T_par, T_biovar, T_tlag>& operator[](const int index) const {
      return Events[index];
    }

    // add lag times
    // template<typename T_parameters, typename T_biovar, typename T_tlag>
    // void AddLagTimes(refactor::ModelParameterHistory<T_time_event, T_parameters, T_biovar, T_tlag> Parameters, int nCmt) // {
    //   int nEvent = Events.size();
    //   int pSize = Parameters.get_size();
    //   assert((pSize = nEvent) || (pSize == 1));

    //   int iEvent = nEvent - 1, evid, cmt, ipar;
    //   PKEvent<T_time_event, T_amt, T_rate, T_ii> newEvent;
    //   while (iEvent >= 0) {
    //     evid = Events[iEvent].evid;
    //     cmt = Events[iEvent].cmt;

    //     if ((evid == 1) || (evid == 4)) {
    //       ipar = std::min(iEvent, pSize - 1);  // ipar is the index of the ith
    //       // event or 0, if the parameters
    //       // are constant.
    //       if (Parameters.GetValueTlag(ipar, cmt - 1) != 0) {
    //         auto newEvent {Events[iEvent]};
    //         newEvent.time += Parameters.GetValueTlag(ipar, cmt - 1);
    //         newEvent.keep = false;
    //         newEvent.isnew = true;
    //         // newEvent.evid = 2  // CHECK
    //         Events.push_back(newEvent);

    //         Events[iEvent].evid = 2;  // Check
    //         // The above statement changes events so that CleanEvents does
    //         // not return an object identical to the original. - CHECK
    //       }
    //     }
    //     iEvent--;
    //   }
    //   sort_by_time();
    // }
    

  };

  template< template<class... > class T, class... Ts>
  struct compare_by_time {
    bool operator()(const T<Ts...> &a, const T<Ts...> &b) {
      return a.time < b.time;
    }
  };

  template< template<class... > class T, class... Ts>
  void sort_by_time(std::vector<T<Ts...> >& v) {
    if (!is_sorted_by_time(v))
      std::stable_sort(v.begin(), v.end(), compare_by_time<T, Ts...>());
  }

  template< template<class... > class T, class... Ts>
  bool is_sorted_by_time(const std::vector<T<Ts...> >& v) {
    return std::is_sorted(v.begin(), v.end(), compare_by_time<T, Ts...>());
  }

}

#endif
