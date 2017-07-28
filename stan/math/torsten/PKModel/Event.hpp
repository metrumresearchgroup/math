#ifndef STAN_MATH_TORSTEN_PKMODEL_EVENT_HPP
#define STAN_MATH_TORSTEN_PKMODEL_EVENT_HPP

#include <Eigen/Dense>
#include <stan/math/torsten/PKModel/functions.hpp>
#include <iostream>
#include <algorithm>
#include <vector>

// forward declare
template <typename T_time, typename T_amt,
  typename T_rate, typename T_ii> class EventHistory;
template <typename T_time, typename T_amt> class RateHistory;
template<typename T_time, typename T_parameters, typename T_biovar,
  typename T_tlag> class ModelParameterHistory;

/**
 * The Event class defines objects that contain the elements of an event,
 * following NONMEM conventions:
 *    time
 *    amt: amount
 *    rate
 *    ii: interdose interval
 *    evid: event identity
 *      (0) observation
 *      (1) dosing
 *      (2) other
 *      (3) reset
 *      (4) reset and dosing
 *    cmt: compartment in which the event occurs
 *    addl: additional doses
 *    ss: steady state approximation (0: no, 1: yes)
 *    keep: if TRUE, save the predicted amount at this event
 *          in the final output of the pred function.
 *    isnew: if TRUE, event was created when pred augmented
 *           the input data set
 */
template <typename T_tau, typename T_amt, typename T_rate, typename T_ii>
class Event{
private:
  T_tau time_;
  T_amt amt_;
  T_rate rate_;
  T_ii ii_;
  int evid_, cmt_, addl_, ss_;
  bool keep_, isnew_;

public:
  Event() { }

  /**
   * Time is passed as a template type. In the augmented
   * event schedule, time depends on tlag, amt, and rate.
   * T_tau should be a promote_args<time, tlag, amt, rate>
   * type, meaning it may be a different type than the base
   * time.
   */
  Event(T_tau time, T_amt amt, T_rate rate, T_ii ii, int evid,
    int cmt, int addl, int ss, bool keep, bool isnew)
    : time_(time), amt_(amt), rate_(rate), ii_(ii), evid_(evid),
      cmt_(cmt), addl_(addl), ss_(ss), keep_(keep), isnew_(isnew) { }

  /**
   * The function operator is handy when we need to define the same event
   * multiple times, as we might in a FOR loop.
   */
  Event operator()(T_tau time, T_amt amt, T_rate rate, T_ii ii,
                   int evid, int cmt, int addl, int ss, bool keep,
                   bool isnew) {
    Event newEvent;
    newEvent.time_ = time;
    newEvent.amt_ = amt;
    newEvent.rate_ = rate;
    newEvent.ii_ = ii;
    newEvent.evid_ = evid;
    newEvent.cmt_ = cmt;
    newEvent.addl_ = addl;
    newEvent.ss_ = ss;
    newEvent.keep_ = keep;
    newEvent.isnew_ = isnew;
    return newEvent;
  }

  // Access functions
  T_tau get_time() { return time_; }
  T_amt get_amt() { return amt_; }
  T_rate get_rate() { return rate_; }
  T_ii get_ii() { return ii_; }
  int get_evid() { return evid_; }
  int get_cmt() { return cmt_; }
  int get_addl() { return addl_; }
  int get_ss() { return ss_; }
  bool get_keep() { return keep_; }
  bool get_isnew() { return isnew_; }

  void Print() {
    std::cout << time_ << " "
              << amt_ << " "
              << rate_ << " "
              << ii_ << " "
              << evid_ << " "
              << cmt_ << " "
              << addl_ << " "
              << ss_ << " "
              << keep_ << " "
              << isnew_ << std::endl;
  }

  // declare friends
  friend class EventHistory<T_tau, T_amt, T_rate, T_ii>;
  template<typename T1, typename T2, typename T3, typename T4> friend
    class ModelParameterHistory;
};

/**
 * The EventHistory class defines objects that contain a std::vector of Events,
 * along with a series of functions that operate on them.
 */
template<typename T_tau, typename T_amt, typename T_rate, typename T_ii>
class EventHistory {
private:
  std::vector<Event<T_tau, T_amt, T_rate, T_ii> > Events_;

public:
  template <typename T_time>
  EventHistory(const std::vector<T_time>& time,
               const std::vector<T_amt>& amt,
               const std::vector<T_rate>& rate,
               const std::vector<T_ii>& ii,
               const std::vector<int>& evid,
               const std::vector<int>& cmt,
               const std::vector<int>& addl,
               const std::vector<int>& ss) {
    Events_.resize(time.size());
    for (size_t i = 0; i < time.size(); i++) {
      Events_[i] = Event<T_tau, T_amt, T_rate, T_ii>(time[i], amt[i], rate[i],
                                                      ii[i], evid[i], cmt[i],
                                                      addl[i], ss[i],
                                                      true, false);
    }
  }

  /**
   * Check if the events are in chronological order
   */
  bool Check() {
    int i = Events_.size() - 1;
    bool ordered = true;

    while ((i > 0) && (ordered)) {
      // note: evid = 3 and evid = 4 correspond to reset events
      ordered = (((Events_[i].time_ >= Events_[i - 1].time_)
        || (Events_[i].evid_ == 3)) || (Events_[i].evid_ == 4));
      i--;
    }
    return ordered;
  }

  Event<T_tau, T_amt, T_rate, T_ii> GetEvent(int i) {
  Event<T_tau, T_amt, T_rate, T_ii>
    newEvent(Events_[i].time_, Events_[i].amt_, Events_[i].rate_, Events_[i].ii_,
      Events_[i].evid_, Events_[i].cmt_, Events_[i].addl_, Events_[i].ss_,
      Events_[i].keep_, Events_[i].isnew_);
    return newEvent;
  }

  void InsertEvent(const Event<T_tau, T_amt, T_rate, T_ii>& Event) {
    Events_.push_back(Event);
  }

  void RemoveEvent(int i) {
    assert(i >= 0);
    Events_.erase(Events_.begin() + i);
  }

  void CleanEvent() {
    for (size_t i = 0; i < Events_.size(); i++)
      if (Events_[i].keep == false) RemoveEvent(i);
   }

  /**
   * Add events to EventHistory object, corresponding to additional dosing,
   * administered at specified inter-dose interval. This information is stored
   * in the addl and ii members of the EventHistory object.
   *
   * Events is sorted at the end of the procedure.
   */
  void AddlDoseEvents() {
    Sort();
    size_t nEvent = Events_.size();
    for (size_t i = 0; i < nEvent; i++) {
      if (((Events_[i].evid_ == 1) || (Events_[i].evid_ == 4))
        && ((Events_[i].addl_ > 0) && (Events_[i].ii_ > 0))) {
        Event<T_tau, T_amt, T_rate, T_ii> addlEvent = GetEvent(i);
        Event<T_tau, T_amt, T_rate, T_ii> newEvent = addlEvent;
        newEvent.addl_ = 0;
        newEvent.ii_ = 0;
        newEvent.ss_ = 0;
        newEvent.keep_ = false;
        newEvent.isnew_ = true;

        for (int j = 1; j <= addlEvent.addl_; j++) {
          newEvent.time_ = addlEvent.time_ + j * addlEvent.ii_;
          InsertEvent(newEvent);
        }
      }
    }
    Sort();
  }

  struct by_time {
    bool operator()(const Event<T_tau, T_amt, T_rate, T_ii>& a,
                    const Event<T_tau, T_amt, T_rate, T_ii>& b) {
        return a.time_ < b.time_;
    }
  };

  void Sort() { std::sort(Events_.begin(), Events_.end(), by_time()); }

  // Access functions
  T_tau get_time(int i) { return Events_[i].time_; }
  T_amt get_amt(int i) { return Events_[i].amt_; }
  T_rate get_rate(int i) { return Events_[i].rate_; }
  T_ii get_ii(int i) { return Events_[i].ii_; }
  int get_evid(int i) { return Events_[i].evid_; }
  int get_cmt(int i) { return Events_[i].cmt_; }
  int get_addl(int i) { return Events_[i].addl_; }
  int get_ss(int i) { return Events_[i].ss_; }
  bool get_keep(int i) { return Events_[i].keep_; }
  bool get_isnew(int i) { return Events_[i].isnew_; }
  int get_size() { return Events_.size(); }

  void Print(int i) {
    std::cout << get_time(i) << " "
              << get_rate(i) << " "
              << get_ii(i) << " "
              << get_evid(i) << " "
              << get_cmt(i) << " "
              << get_addl(i) << " "
              << get_ss(i) << " "
              << get_keep(i) << " "
              << get_isnew(i) << std::endl;
    }

  /**
   * Implement absorption lag times by modifying the times of the dosing events.
   * Two cases: parameters are either constant or vary with each event.
   * Function sorts events at the end of the procedure.
   *
   * @tparam T_parameters type of scalar model parameters
   * @param[in] ModelParameterHistory object that stores parameters for each event
   * @param[in] nCmt
   * @return - modified events that account for absorption lag times
   */
  template<typename T_parameters, typename T_biovar, typename T_tlag>
  void AddLagTimes(ModelParameterHistory<T_tau, T_parameters, T_biovar,
                   T_tlag> Parameters, int nCmt) {
    int nEvent = Events_.size(), pSize = Parameters.get_size();
    assert((pSize = nEvent) || (pSize == 1));

    int iEvent = nEvent - 1, evid, cmt, ipar;
    Event<T_tau, T_amt, T_rate, T_ii> newEvent;
    while (iEvent >= 0) {
      evid = Events_[iEvent].evid_;
      cmt = Events_[iEvent].cmt_;

      if ((evid == 1) || (evid == 4)) {
        ipar = std::min(iEvent, pSize - 1);  // ipar is the index of the ith
                                             // event or 0, if the parameters
                                             // are constant.
        if (Parameters.GetValueTlag(ipar, cmt - 1) != 0) {
          newEvent = GetEvent(iEvent);
          newEvent.time_ += Parameters.GetValueTlag(ipar, cmt - 1);
          newEvent.keep_ = false;
          newEvent.isnew_ = true;
          // newEvent.evid = 2  // CHECK
          InsertEvent(newEvent);

          Events_[iEvent].evid_ = 2;  // Check
          // The above statement changes events so that CleanEvents does
          // not return an object identical to the original. - CHECK
        }
      }
      iEvent--;
    }
    Sort();
  }

  // declare friends
  friend class Event<T_tau, T_amt, T_rate, T_ii>;
  template<typename T1, typename T2, typename T3, typename T4> friend
    class ModelParameterHistory;
};

#endif
