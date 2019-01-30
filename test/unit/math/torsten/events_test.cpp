#include <stan/math/rev/mat.hpp>  // FIX ME - includes should be more specific
#include <test/unit/math/torsten/expect_near_matrix_eq.hpp>
#include <test/unit/math/torsten/expect_matrix_eq.hpp>
#include <test/unit/math/torsten/pk_twocpt_test_fixture.hpp>
#include <test/unit/math/torsten/test_util.hpp>
#include <stan/math/torsten/PKModelTwoCpt.hpp>
#include <stan/math/torsten/pk_onecpt_model.hpp>
#include <stan/math/torsten/pk_twocpt_model.hpp>
#include <gtest/gtest.h>
#include <vector>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;
using stan::math::var;
using refactor::PKRec;
using torsten::EventsManager;

TEST_F(TorstenPKTwoCptTest, events_addl) {
  using EM = EventsManager<double, double, double, double, double, double, double>;

  int nCmt = refactor::PKTwoCptModel<double, double, double, double>::Ncmt;
  addl[0] = 5;
  addl[3] = 3;

  {
    EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);
    auto ev = em.events();
    EXPECT_EQ(ev.size(), evid.size() + addl[0]);
  }

  {
    ii[3] = 4.0;
    EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);
    auto ev = em.events();
    EXPECT_EQ(ev.size(), evid.size() + addl[0]);
  }

  amt[3] = 400.0;
  evid[3] = 1;
  EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);
  auto ev = em.events();
  EXPECT_EQ(ev.size(), evid.size() + addl[0] + addl[3]);

  EXPECT_EQ(ev.time(0 ), 0    );  EXPECT_EQ(ev.amt(0 ), 1000);  EXPECT_EQ(ev.evid(0 ), 1);
  EXPECT_EQ(ev.time(1 ), 0.25 );  EXPECT_EQ(ev.amt(1 ), 0   );  EXPECT_EQ(ev.evid(1 ), 0);
  EXPECT_EQ(ev.time(2 ), 0.5  );  EXPECT_EQ(ev.amt(2 ), 0   );  EXPECT_EQ(ev.evid(2 ), 0);
  EXPECT_EQ(ev.time(3 ), 0.75 );  EXPECT_EQ(ev.amt(3 ), 400 );  EXPECT_EQ(ev.evid(3 ), 1);
  EXPECT_EQ(ev.time(4 ), 1    );  EXPECT_EQ(ev.amt(4 ), 0   );  EXPECT_EQ(ev.evid(4 ), 0);
  EXPECT_EQ(ev.time(5 ), 1.25 );  EXPECT_EQ(ev.amt(5 ), 0   );  EXPECT_EQ(ev.evid(5 ), 0);
  EXPECT_EQ(ev.time(6 ), 1.5  );  EXPECT_EQ(ev.amt(6 ), 0   );  EXPECT_EQ(ev.evid(6 ), 0);
  EXPECT_EQ(ev.time(7 ), 1.75 );  EXPECT_EQ(ev.amt(7 ), 0   );  EXPECT_EQ(ev.evid(7 ), 0);
  EXPECT_EQ(ev.time(8 ), 2    );  EXPECT_EQ(ev.amt(8 ), 0   );  EXPECT_EQ(ev.evid(8 ), 0);
  EXPECT_EQ(ev.time(9 ), 4    );  EXPECT_EQ(ev.amt(9 ), 0   );  EXPECT_EQ(ev.evid(9 ), 0);
  EXPECT_EQ(ev.time(10), 4.75 );  EXPECT_EQ(ev.amt(10), 400 );  EXPECT_EQ(ev.evid(10), 1);
  EXPECT_EQ(ev.time(11), 8.75 );  EXPECT_EQ(ev.amt(11), 400 );  EXPECT_EQ(ev.evid(11), 1);
  EXPECT_EQ(ev.time(12), 12   );  EXPECT_EQ(ev.amt(12), 1000);  EXPECT_EQ(ev.evid(12), 1);
  EXPECT_EQ(ev.time(13), 12.75);  EXPECT_EQ(ev.amt(13), 400 );  EXPECT_EQ(ev.evid(13), 1);
  EXPECT_EQ(ev.time(14), 24   );  EXPECT_EQ(ev.amt(14), 1000);  EXPECT_EQ(ev.evid(14), 1);
  EXPECT_EQ(ev.time(15), 36   );  EXPECT_EQ(ev.amt(15), 1000);  EXPECT_EQ(ev.evid(15), 1);
  EXPECT_EQ(ev.time(16), 48   );  EXPECT_EQ(ev.amt(16), 1000);  EXPECT_EQ(ev.evid(16), 1);
  EXPECT_EQ(ev.time(17), 60   );  EXPECT_EQ(ev.amt(17), 1000);  EXPECT_EQ(ev.evid(17), 1);
}

TEST_F(TorstenPKTwoCptTest, events_addl_rate) {
  using EM = EventsManager<double, double, double, double, double, double, double>;

  int nCmt = refactor::PKTwoCptModel<double, double, double, double>::Ncmt;
  addl[0] = 5;
  addl[3] = 2;
  amt[3] = 1200.0;
  ii[3] = 5;
  cmt[3] = 2;
  rate[3] = 400;
  evid[3] = 1;

  EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);
  auto ev = em.events();

  /* each IV dose has an end event.*/
  EXPECT_EQ(ev.size(), evid.size() + addl[0] + addl[3] * 2 + 1);

  EXPECT_FLOAT_EQ(ev.time(0 ), 0    );   EXPECT_EQ(ev.evid(0 ), 1);
  EXPECT_FLOAT_EQ(ev.time(1 ), 0.25 );   EXPECT_EQ(ev.evid(1 ), 0);
  EXPECT_FLOAT_EQ(ev.time(2 ), 0.5  );   EXPECT_EQ(ev.evid(2 ), 0);
  EXPECT_FLOAT_EQ(ev.time(3 ), 0.75 );   EXPECT_EQ(ev.evid(3 ), 1);
  EXPECT_FLOAT_EQ(ev.time(4 ), 1    );   EXPECT_EQ(ev.evid(4 ), 0);
  EXPECT_FLOAT_EQ(ev.time(5 ), 1.25 );   EXPECT_EQ(ev.evid(5 ), 0);
  EXPECT_FLOAT_EQ(ev.time(6 ), 1.5  );   EXPECT_EQ(ev.evid(6 ), 0);
  EXPECT_FLOAT_EQ(ev.time(7 ), 1.75 );   EXPECT_EQ(ev.evid(7 ), 0);
  EXPECT_FLOAT_EQ(ev.time(8 ), 2    );   EXPECT_EQ(ev.evid(8 ), 0);
  EXPECT_FLOAT_EQ(ev.time(9 ), 3.75 );   EXPECT_EQ(ev.evid(9 ), 2);
  EXPECT_FLOAT_EQ(ev.time(10), 4    );   EXPECT_EQ(ev.evid(10), 0);
  EXPECT_FLOAT_EQ(ev.time(11), 5.75 );   EXPECT_EQ(ev.evid(11), 1);
  EXPECT_FLOAT_EQ(ev.time(12), 8.75 );   EXPECT_EQ(ev.evid(12), 2);
  EXPECT_FLOAT_EQ(ev.time(13), 10.75);   EXPECT_EQ(ev.evid(13), 1);
  EXPECT_FLOAT_EQ(ev.time(14), 12   );   EXPECT_EQ(ev.evid(14), 1);
  EXPECT_FLOAT_EQ(ev.time(15), 13.75);   EXPECT_EQ(ev.evid(15), 2);
  EXPECT_FLOAT_EQ(ev.time(16), 24   );   EXPECT_EQ(ev.evid(16), 1);
  EXPECT_FLOAT_EQ(ev.time(17), 36   );   EXPECT_EQ(ev.evid(17), 1);
  EXPECT_FLOAT_EQ(ev.time(18), 48   );   EXPECT_EQ(ev.evid(18), 1);
  EXPECT_FLOAT_EQ(ev.time(19), 60   );   EXPECT_EQ(ev.evid(19), 1);

  EXPECT_FLOAT_EQ(em.rates()[0 ][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[1 ][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[2 ][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[3 ][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[4 ][1], 400);
  EXPECT_FLOAT_EQ(em.rates()[5 ][1], 400);
  EXPECT_FLOAT_EQ(em.rates()[6 ][1], 400);
  EXPECT_FLOAT_EQ(em.rates()[7 ][1], 400);
  EXPECT_FLOAT_EQ(em.rates()[8 ][1], 400);
  EXPECT_FLOAT_EQ(em.rates()[9 ][1], 400);
  EXPECT_FLOAT_EQ(em.rates()[10][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[11][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[12][1], 400);
  EXPECT_FLOAT_EQ(em.rates()[13][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[14][1], 400);
  EXPECT_FLOAT_EQ(em.rates()[15][1], 400);
  EXPECT_FLOAT_EQ(em.rates()[16][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[17][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[18][1], 0  );
  EXPECT_FLOAT_EQ(em.rates()[19][1], 0  );
}

TEST_F(TorstenPKTwoCptTest, events_addl_tlag) {
  using EM = EventsManager<double, double, double, double, double, double, double>;

  int nCmt = refactor::PKTwoCptModel<double, double, double, double>::Ncmt;
  addl[0] = 1;
  addl[3] = 2;
  amt[3] = 1200.0;
  ii[3] = 3;
  cmt[3] = 2;
  evid[3] = 4;

  // lag for cmt 1, this shouldn't affect the dose on cmt 2.
  tlag[0][0] = 0.25;
  {
    EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);
    auto ev = em.events();
    EXPECT_EQ(ev.size(), time.size() + addl[0] * 2 + 1 + addl[3]);

    EXPECT_FLOAT_EQ(ev.time(0 ), 0    ); EXPECT_FLOAT_EQ(ev.amt(0 ), 1000.); 
    EXPECT_FLOAT_EQ(ev.time(1 ), 0.25 ); EXPECT_FLOAT_EQ(ev.amt(1 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(2 ), 0.25 ); EXPECT_FLOAT_EQ(ev.amt(2 ), 1000.); 
    EXPECT_FLOAT_EQ(ev.time(3 ), 0.5  ); EXPECT_FLOAT_EQ(ev.amt(3 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(4 ), 0.75 ); EXPECT_FLOAT_EQ(ev.amt(4 ), 1200.); 
    EXPECT_FLOAT_EQ(ev.time(5 ), 1    ); EXPECT_FLOAT_EQ(ev.amt(5 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(6 ), 1.25 ); EXPECT_FLOAT_EQ(ev.amt(6 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(7 ), 1.5  ); EXPECT_FLOAT_EQ(ev.amt(7 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(8 ), 1.75 ); EXPECT_FLOAT_EQ(ev.amt(8 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(9 ), 2    ); EXPECT_FLOAT_EQ(ev.amt(9 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(10), 3.75 ); EXPECT_FLOAT_EQ(ev.amt(10), 1200.); 
    EXPECT_FLOAT_EQ(ev.time(11), 4    ); EXPECT_FLOAT_EQ(ev.amt(11), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(12), 6.75 ); EXPECT_FLOAT_EQ(ev.amt(12), 1200.); 
    EXPECT_FLOAT_EQ(ev.time(13), 12   ); EXPECT_FLOAT_EQ(ev.amt(13), 1000.); 
    EXPECT_FLOAT_EQ(ev.time(14), 12.25); EXPECT_FLOAT_EQ(ev.amt(14), 1000.); 

    EXPECT_EQ(ev.cmt(0 ), 1);     EXPECT_EQ(ev.evid(0 ), 2);
    EXPECT_EQ(ev.cmt(1 ), 3);     EXPECT_EQ(ev.evid(1 ), 0);
    EXPECT_EQ(ev.cmt(2 ), 1);     EXPECT_EQ(ev.evid(2 ), 1);
    EXPECT_EQ(ev.cmt(3 ), 3);     EXPECT_EQ(ev.evid(3 ), 0);
    EXPECT_EQ(ev.cmt(4 ), 2);     EXPECT_EQ(ev.evid(4 ), 4);
    EXPECT_EQ(ev.cmt(5 ), 3);     EXPECT_EQ(ev.evid(5 ), 0);
    EXPECT_EQ(ev.cmt(6 ), 3);     EXPECT_EQ(ev.evid(6 ), 0);
    EXPECT_EQ(ev.cmt(7 ), 3);     EXPECT_EQ(ev.evid(7 ), 0);
    EXPECT_EQ(ev.cmt(8 ), 3);     EXPECT_EQ(ev.evid(8 ), 0);
    EXPECT_EQ(ev.cmt(9 ), 3);     EXPECT_EQ(ev.evid(9 ), 0);
    EXPECT_EQ(ev.cmt(10), 2);     EXPECT_EQ(ev.evid(10), 4);
    EXPECT_EQ(ev.cmt(11), 3);     EXPECT_EQ(ev.evid(11), 0);
    EXPECT_EQ(ev.cmt(12), 2);     EXPECT_EQ(ev.evid(12), 4);
    EXPECT_EQ(ev.cmt(13), 1);     EXPECT_EQ(ev.evid(13), 2);
    EXPECT_EQ(ev.cmt(14), 1);     EXPECT_EQ(ev.evid(14), 1);
  }

  // lag for cmt 2, this shouldn't affect the dose on cmt 1.
  tlag[0][0] = 0.0;
  tlag[0][1] = 0.2;
  {
    EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);
    auto ev = em.events();
    EXPECT_EQ(ev.size(), time.size() + addl[3] * 2 + 1 + addl[0]);
    
    EXPECT_FLOAT_EQ(ev.time(0 ), 0   );   EXPECT_FLOAT_EQ(ev.amt(0 ), 1000.); 
    EXPECT_FLOAT_EQ(ev.time(1 ), 0.25);   EXPECT_FLOAT_EQ(ev.amt(1 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(2 ), 0.5 );   EXPECT_FLOAT_EQ(ev.amt(2 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(3 ), 0.75);   EXPECT_FLOAT_EQ(ev.amt(3 ), 1200.); 
    EXPECT_FLOAT_EQ(ev.time(4 ), 0.95);   EXPECT_FLOAT_EQ(ev.amt(4 ), 1200.); 
    EXPECT_FLOAT_EQ(ev.time(5 ), 1   );   EXPECT_FLOAT_EQ(ev.amt(5 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(6 ), 1.25);   EXPECT_FLOAT_EQ(ev.amt(6 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(7 ), 1.5 );   EXPECT_FLOAT_EQ(ev.amt(7 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(8 ), 1.75);   EXPECT_FLOAT_EQ(ev.amt(8 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(9 ), 2   );   EXPECT_FLOAT_EQ(ev.amt(9 ), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(10), 3.75);   EXPECT_FLOAT_EQ(ev.amt(10), 1200.); 
    EXPECT_FLOAT_EQ(ev.time(11), 3.95);   EXPECT_FLOAT_EQ(ev.amt(11), 1200.); 
    EXPECT_FLOAT_EQ(ev.time(12), 4   );   EXPECT_FLOAT_EQ(ev.amt(12), 0.   ); 
    EXPECT_FLOAT_EQ(ev.time(13), 6.75);   EXPECT_FLOAT_EQ(ev.amt(13), 1200.); 
    EXPECT_FLOAT_EQ(ev.time(14), 6.95);   EXPECT_FLOAT_EQ(ev.amt(14), 1200.); 
    EXPECT_FLOAT_EQ(ev.time(15), 12  );   EXPECT_FLOAT_EQ(ev.amt(15), 1000.); 

    EXPECT_EQ(ev.cmt(0 ), 1);     EXPECT_EQ(ev.evid(0 ), 1);
    EXPECT_EQ(ev.cmt(1 ), 3);     EXPECT_EQ(ev.evid(1 ), 0);
    EXPECT_EQ(ev.cmt(2 ), 3);     EXPECT_EQ(ev.evid(2 ), 0);
    EXPECT_EQ(ev.cmt(3 ), 2);     EXPECT_EQ(ev.evid(3 ), 2);
    EXPECT_EQ(ev.cmt(4 ), 2);     EXPECT_EQ(ev.evid(4 ), 4);
    EXPECT_EQ(ev.cmt(5 ), 3);     EXPECT_EQ(ev.evid(5 ), 0);
    EXPECT_EQ(ev.cmt(6 ), 3);     EXPECT_EQ(ev.evid(6 ), 0);
    EXPECT_EQ(ev.cmt(7 ), 3);     EXPECT_EQ(ev.evid(7 ), 0);
    EXPECT_EQ(ev.cmt(8 ), 3);     EXPECT_EQ(ev.evid(8 ), 0);
    EXPECT_EQ(ev.cmt(9 ), 3);     EXPECT_EQ(ev.evid(9 ), 0);
    EXPECT_EQ(ev.cmt(10), 2);     EXPECT_EQ(ev.evid(10), 2);
    EXPECT_EQ(ev.cmt(11), 2);     EXPECT_EQ(ev.evid(11), 4);
    EXPECT_EQ(ev.cmt(12), 3);     EXPECT_EQ(ev.evid(12), 0);
    EXPECT_EQ(ev.cmt(13), 2);     EXPECT_EQ(ev.evid(13), 2);
    EXPECT_EQ(ev.cmt(14), 2);     EXPECT_EQ(ev.evid(14), 4);
    EXPECT_EQ(ev.cmt(15), 1);     EXPECT_EQ(ev.evid(15), 1);
    }
}

TEST_F(TorstenPKTwoCptTest, events_addl_rate_const_tlag) {
  using EM = EventsManager<double, double, double, double, double, double, double>;

  int nCmt = refactor::PKTwoCptModel<double, double, double, double>::Ncmt;
  addl[0] = 1;
  addl[3] = 2;
  amt[3] = 1200.0;
  ii[3] = 5;
  cmt[3] = 2;
  rate[3] = 400;
  evid[3] = 1;

  tlag[0][1] = 0.25;
  EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);
  auto ev = em.events();
  EXPECT_EQ(ev.size(), time.size() + addl[3] * 3 + 2 + addl[0]);

    EXPECT_FLOAT_EQ(ev.time(0 ), 0    ); EXPECT_FLOAT_EQ(em.rates()[0 ][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(1 ), 0.25 ); EXPECT_FLOAT_EQ(em.rates()[1 ][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(2 ), 0.5  ); EXPECT_FLOAT_EQ(em.rates()[2 ][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(3 ), 0.75 ); EXPECT_FLOAT_EQ(em.rates()[3 ][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(4 ), 1    ); EXPECT_FLOAT_EQ(em.rates()[4 ][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(5 ), 1    ); EXPECT_FLOAT_EQ(em.rates()[5 ][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(6 ), 1.25 ); EXPECT_FLOAT_EQ(em.rates()[6 ][1], 400.); 
    EXPECT_FLOAT_EQ(ev.time(7 ), 1.5  ); EXPECT_FLOAT_EQ(em.rates()[7 ][1], 400.); 
    EXPECT_FLOAT_EQ(ev.time(8 ), 1.75 ); EXPECT_FLOAT_EQ(em.rates()[8 ][1], 400.); 
    EXPECT_FLOAT_EQ(ev.time(9 ), 2    ); EXPECT_FLOAT_EQ(em.rates()[9 ][1], 400.); 
    EXPECT_FLOAT_EQ(ev.time(10), 4    ); EXPECT_FLOAT_EQ(em.rates()[10][1], 400.); 
    EXPECT_FLOAT_EQ(ev.time(11), 4    ); EXPECT_FLOAT_EQ(em.rates()[11][1], 400.); 
    EXPECT_FLOAT_EQ(ev.time(12), 5.75 ); EXPECT_FLOAT_EQ(em.rates()[12][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(13), 6    ); EXPECT_FLOAT_EQ(em.rates()[13][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(14), 9    ); EXPECT_FLOAT_EQ(em.rates()[14][1], 400.); 
    EXPECT_FLOAT_EQ(ev.time(15), 10.75); EXPECT_FLOAT_EQ(em.rates()[15][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(16), 11   ); EXPECT_FLOAT_EQ(em.rates()[16][1], 0.  ); 
    EXPECT_FLOAT_EQ(ev.time(17), 12   ); EXPECT_FLOAT_EQ(em.rates()[17][1], 400.); 
    EXPECT_FLOAT_EQ(ev.time(18), 14   ); EXPECT_FLOAT_EQ(em.rates()[18][1], 400.); 

    EXPECT_EQ(ev.evid(0 ), 1);  EXPECT_EQ(ev.cmt(0 ), 1);
    EXPECT_EQ(ev.evid(1 ), 0);  EXPECT_EQ(ev.cmt(1 ), 3);
    EXPECT_EQ(ev.evid(2 ), 0);  EXPECT_EQ(ev.cmt(2 ), 3);
    EXPECT_EQ(ev.evid(3 ), 2);  EXPECT_EQ(ev.cmt(3 ), 2);
    EXPECT_EQ(ev.evid(4 ), 0);  EXPECT_EQ(ev.cmt(4 ), 3);
    EXPECT_EQ(ev.evid(5 ), 1);  EXPECT_EQ(ev.cmt(5 ), 2);
    EXPECT_EQ(ev.evid(6 ), 0);  EXPECT_EQ(ev.cmt(6 ), 3);
    EXPECT_EQ(ev.evid(7 ), 0);  EXPECT_EQ(ev.cmt(7 ), 3);
    EXPECT_EQ(ev.evid(8 ), 0);  EXPECT_EQ(ev.cmt(8 ), 3);
    EXPECT_EQ(ev.evid(9 ), 0);  EXPECT_EQ(ev.cmt(9 ), 3);
    EXPECT_EQ(ev.evid(10), 0);  EXPECT_EQ(ev.cmt(10), 3);
    EXPECT_EQ(ev.evid(11), 2);  EXPECT_EQ(ev.cmt(11), 2);
    EXPECT_EQ(ev.evid(12), 2);  EXPECT_EQ(ev.cmt(12), 2);
    EXPECT_EQ(ev.evid(13), 1);  EXPECT_EQ(ev.cmt(13), 2);
    EXPECT_EQ(ev.evid(14), 2);  EXPECT_EQ(ev.cmt(14), 2);
    EXPECT_EQ(ev.evid(15), 2);  EXPECT_EQ(ev.cmt(15), 2);
    EXPECT_EQ(ev.evid(16), 1);  EXPECT_EQ(ev.cmt(16), 2);
    EXPECT_EQ(ev.evid(17), 1);  EXPECT_EQ(ev.cmt(17), 1);
    EXPECT_EQ(ev.evid(18), 2);  EXPECT_EQ(ev.cmt(18), 2);
}

TEST_F(TorstenPKTwoCptTest, events_addl_rate_tlag) {
  using EM = EventsManager<double, double, double, double, double, double, double>;

  int nCmt = refactor::PKTwoCptModel<double, double, double, double>::Ncmt;
  addl[0] = 0;
  addl[3] = 2;
  amt[3] = 1200.0;
  ii[3] = 5;
  cmt[3] = 2;
  rate[3] = 400;
  evid[3] = 1;

  tlag.resize(time.size());
  for (auto& l : tlag) l.resize(nCmt);
  tlag[3][1] = 0.25;

  EM em(nCmt, time, amt, rate, ii, evid, cmt, addl, ss, pMatrix, biovar, tlag);
  auto ev = em.events();

  // time-depdent tlag has not effect on new events
  EXPECT_EQ(ev.size(), time.size() + addl[3] * 2 + 2);

  EXPECT_FLOAT_EQ(ev.time(0 ), 0    ); EXPECT_FLOAT_EQ(em.rates()[0 ][1],    0.0);
  EXPECT_FLOAT_EQ(ev.time(1 ), 0.25 ); EXPECT_FLOAT_EQ(em.rates()[1 ][1],    0.0);
  EXPECT_FLOAT_EQ(ev.time(2 ), 0.5  ); EXPECT_FLOAT_EQ(em.rates()[2 ][1],    0.0);
  EXPECT_FLOAT_EQ(ev.time(3 ), 0.75 ); EXPECT_FLOAT_EQ(em.rates()[3 ][1],    0.0);
  EXPECT_FLOAT_EQ(ev.time(4 ), 1    ); EXPECT_FLOAT_EQ(em.rates()[4 ][1],    0.0);
  EXPECT_FLOAT_EQ(ev.time(5 ), 1    ); EXPECT_FLOAT_EQ(em.rates()[5 ][1],    0.0);
  EXPECT_FLOAT_EQ(ev.time(6 ), 1.25 ); EXPECT_FLOAT_EQ(em.rates()[6 ][1],  400.0);
  EXPECT_FLOAT_EQ(ev.time(7 ), 1.5  ); EXPECT_FLOAT_EQ(em.rates()[7 ][1],  400.0);
  EXPECT_FLOAT_EQ(ev.time(8 ), 1.75 ); EXPECT_FLOAT_EQ(em.rates()[8 ][1],  400.0);
  EXPECT_FLOAT_EQ(ev.time(9 ), 2    ); EXPECT_FLOAT_EQ(em.rates()[9 ][1],  400.0);
  EXPECT_FLOAT_EQ(ev.time(10), 4    ); EXPECT_FLOAT_EQ(em.rates()[10][1],  400.0);
  EXPECT_FLOAT_EQ(ev.time(11), 4    ); EXPECT_FLOAT_EQ(em.rates()[11][1],  400.0);
  EXPECT_FLOAT_EQ(ev.time(12), 5.75 ); EXPECT_FLOAT_EQ(em.rates()[12][1],    0.0);
  EXPECT_FLOAT_EQ(ev.time(13), 8.75 ); EXPECT_FLOAT_EQ(em.rates()[13][1],  400.0);
  EXPECT_FLOAT_EQ(ev.time(14), 10.75); EXPECT_FLOAT_EQ(em.rates()[14][1],    0.0);
  EXPECT_FLOAT_EQ(ev.time(15), 13.75); EXPECT_FLOAT_EQ(em.rates()[15][1],  400.0);
}
