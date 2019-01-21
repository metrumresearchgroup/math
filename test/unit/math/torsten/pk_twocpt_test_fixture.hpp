#ifndef TEST_UNIT_TORSTEN_PK_TWOCPT_MODEL_TEST_FIXTURE
#define TEST_UNIT_TORSTEN_PK_TWOCPT_MODEL_TEST_FIXTURE

#include <stan/math/rev/mat.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <string>

class TorstenPKTwoCptTest : public testing::Test {
  void SetUp() {
    // make sure memory's clean before starting each test
    stan::math::recover_memory();
  }
public:
  TorstenPKTwoCptTest() :
    nt(10),
    time(nt, 0.0),
    amt(nt, 0),
    rate(nt, 0),
    cmt(nt, 3),
    evid(nt, 0),
    ii(nt, 0),
    addl(nt, 0),
    ss(nt, 0),
    pMatrix{ {5, 8, 20, 70, 1.2 } },
    biovar{ { 1, 1, 1 } },
    tlag{ { 0, 0, 0 } },
    np(200),
    time_m   (np),
    amt_m    (np),
    rate_m   (np),
    cmt_m    (np),
    evid_m   (np),
    ii_m     (np),
    addl_m   (np),
    ss_m     (np),
    pMatrix_m(np, pMatrix),
    biovar_m (np, biovar),
    tlag_m   (np, tlag)
  {
    torsten::mpi::init();

    time[0] = 0;
    for(int i = 1; i < 9; i++) time[i] = time[i - 1] + 0.25;
    time[9] = 4.0;
    amt[0] = 1000;
    cmt[0] = 1;    
    evid[0] = 1;
    ii[0] = 12;
    addl[0] = 14;

    // population data
    for (int i = 0; i < np; ++i) {
      time_m[i] = time;
      amt_m [i] = amt; 
      rate_m[i] = rate;
      cmt_m [i] = cmt; 
      evid_m[i] = evid;
      ii_m  [i] = ii;  
      addl_m[i] = addl;
      ss_m  [i] = ss;  
    }

    SetUp();
  }

  const int nt;
  std::vector<double> time;
  std::vector<double> amt;
  std::vector<double> rate;
  std::vector<int> cmt;
  std::vector<int> evid;
  std::vector<double> ii;
  std::vector<int> addl;
  std::vector<int> ss;
  std::vector<std::vector<double> > pMatrix;  // CL, VC, Ka
  std::vector<std::vector<double> > biovar;
  std::vector<std::vector<double> > tlag;
  const int np;
  std::vector<std::vector<double        > >   time_m   ;
  std::vector<std::vector<double        > >   amt_m    ;
  std::vector<std::vector<double        > >   rate_m   ;
  std::vector<std::vector<int           > >   cmt_m    ;
  std::vector<std::vector<int           > >   evid_m   ;
  std::vector<std::vector<double        > >   ii_m     ;
  std::vector<std::vector<int           > >   addl_m   ;
  std::vector<std::vector<int           > >   ss_m     ;
  std::vector<std::vector<std::vector<double > > > pMatrix_m;
  std::vector<std::vector<std::vector<double > > > biovar_m ;
  std::vector<std::vector<std::vector<double > > > tlag_m   ;
};

#endif
