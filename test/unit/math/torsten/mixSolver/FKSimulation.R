## Template to simulate PKPD data
rm(list = ls())
gc()

library(mrgsolve)
library(rstan)
library(dplyr)

modelName <- "neutropeniaSingle"

code <- '
$PARAM CL = 10, Q = 15, VC = 35, VP = 105, KA = 2.0,
MTT = 125, Circ0 = 5, alpha = 3E-4, gamma = 0.17

$SET delta=0.1 // simulation grid

$CMT GUT CENT PERI PROL TRANSIT1 TRANSIT2 TRANSIT3 CIRC

$MAIN

// Reparametrization
double k10 = CL / VC;
double k12 = Q / VC;
double k21 = Q / VP;
double ktr = 4 / MTT;

$ODE 
dxdt_GUT = -KA * GUT;
dxdt_CENT = KA * GUT - (k10 + k12) * CENT + k21 * PERI;
dxdt_PERI = k12 * CENT - k21 * PERI;
dxdt_PROL = ktr * (PROL + Circ0) * ((1 - alpha * CENT/VC) * pow(Circ0/(CIRC + Circ0),gamma) - 1);
dxdt_TRANSIT1 = ktr * (PROL - TRANSIT1);
dxdt_TRANSIT2 = ktr * (TRANSIT1 - TRANSIT2);
dxdt_TRANSIT3 = ktr * (TRANSIT2 - TRANSIT3);
dxdt_CIRC = ktr * (TRANSIT3 - CIRC);

$SIGMA 0.001 0.001 

$TABLE
double CP = CENT/VC;
double DV1 = CENT/VC * exp(EPS(1));
double DV2 = (CIRC + Circ0) * exp(EPS(2));

$CAPTURE CP DV1 DV2
'

mod <- mread("acum", tempdir(), code)
e1 <- ev(amt = 80 * 1000, ii = 12, addl = 14) # Create dosing events
mod %>% ev(e1) %>% mrgsim(end = 500) %>% plot # plot data

## Observation and dosing times
# doseTimes <- seq(0, 168, by = 12)
xpk <- c(0, 0.083, 0.167, 0.25, 0.5, 0.75, 1, 1.5, 2,3,4,6,8)
time <- sort(unique(xpk))

# save data in data frame 
xdata <- 
  mod %>% 
  ev(e1) %>% 
  carry_out(cmt, ii, addl, rate, amt, evid, ss) %>%
  mrgsim(Req = "GUT, CENT, PERI, PROL, TRANSIT1, TRANSIT2, TRANSIT3, CIRC", end = -1, add = time, recsort = 3) %>%
  as.data.frame

xdata
