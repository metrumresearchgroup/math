## FIX ME - Put simulations for all tests in PKModelOneCpt_test.cpp
## in this file.

## Simulate solutions for unit tests in Torsten
rm(list = ls())
gc()

# .libPaths("~/svn-StanPmetrics/script/lib")
library(mrgsolve)
library(dplyr)

## Simulate for One Compartment Model
code <- '
$PARAM CL = 5, Q = 8, V2 = 35, V3 = 105, KA = 1.2
$CMT GUT CENT PERI
$GLOBAL
#define CP (CENT/V2)

$PKMODEL ncmt = 2, depot = TRUE
$SIGMA 0.01  // variance
$TABLE
capture DV = CP * exp(EPS(1));
$CAPTURE CP
'

mod <- mread("acum", tempdir(), code)
e1 <- ev(amt = 1200, rate = 1200, addl = 14, ii = 12)
mod %>% ev(e1) %>% mrgsim(end = 500) %>% plot # plot data

## save some data for unit tests (see amounts at t = 1 hour, with no noise)
time <- seq(from = 0.25, to = 2, by = 0.25)
time <- c(time, 4)
xdata <- mod %>% ev(e1) %>% mrgsim(Req = "GUT, CENT, PERI",
                                   end = -1, add = time,
                                   rescort = 3) %>% as.data.frame
xdata
# ID time       GUT      CENT        PERI
# 1   1 0.00   0.00000   0.00000   0.0000000
# 2   1 0.25 259.18178  39.55748   0.7743944
# 3   1 0.50 451.18836 139.65573   5.6130073
# 4   1 0.75 593.43034 278.43884  17.2109885
# 5   1 1.00 698.80579 440.32663  37.1629388
# 6   1 1.25 517.68806 574.76950  65.5141658
# 7   1 1.50 383.51275 653.13596  99.2568509
# 8   1 1.75 284.11323 692.06145 135.6122367
# 9   1 2.00 210.47626 703.65965 172.6607082
# 10  1 4.00  19.09398 486.11014 406.6342765

## Steady state with multiple truncated infusion
time <- seq(from = 0, to = 45, by = 5)
e1 <- ev(amt = 1200, rate = 150, ii = 12, ss = 1)
xdata <- mod %>% ev(e1) %>% mrgsim(Req = "GUT, CENT, PERI",
                                   end = -1, add = time,
                                   rescort = 3) %>% as.data.frame
xdata
# ID time          GUT     CENT      PERI
# 1   1    0 0.000000e+00   0.0000    0.0000
# 2   1    0 1.028649e+00 562.0698 2109.5917
# 3   1    5 1.246927e+02 758.1440 2071.4754
# 4   1   10 1.133898e+01 686.1621 2152.0095
# 5   1   15 2.810653e-02 466.1906 1988.7904
# 6   1   20 6.966912e-05 391.4036 1760.2212
# 7   1   25 1.726925e-07 341.7953 1548.6370
# 8   1   30 4.280618e-10 300.1529 1361.3663
# 9   1   35 1.061059e-12 263.7890 1196.6058
# 10  1   40 2.630103e-15 231.8555 1051.7691
# 11  1   45 6.519373e-18 203.7908  924.4614

