## Analyze CPU times for mix and full integration solvers
## NOTE: don't trust comments, need to rerun this properly

rm(list = ls())
gc()

.libPaths("~/svn-StanPmetrics/script/lib")
setwd("~/Desktop/Code/torsten/mixSolver/math/test/unit/math/torsten/mixSolver")
library(ggplot2)

##########################################################################
## Friberg-Karlsson Model: dv population regime
## Pass initial states as fixed data
## Pass parameters as parameters
## One set of initial estimates and parameters 
data <- read.csv(file = "mixSolverResult_pop_dv.csv")
data$X <- NULL
data$X0 <- NULL
N <- length(head(data))  ## expect 100

## summary plots
# 1: plot ratio against number of parameters (9 * number of "patients")
Nmax <- nrow(data) / 2;
data_num <- rep(0, N)
data_mix <- rep(0, N)
ratios <- rep(0, Nmax)

## FIX ME - may be a way to do this w/o for loop
for (j in 0:(Nmax - 1)) {
  for (i in 1:N) { 
    data_num[i] <- data[2 * j + 1, i]
    data_mix[i] <- data[2 * j + 2, i]
  }
  ratios[j + 1] <- mean(data_mix) / mean(data_num)
}

## very basic plot
plot(seq(1:Nmax), ratios)


# 2: plot box plots to show spread of ratios
## FIX ME - code w/o for loop
boxData <- data.frame()
RUN <- rep(factor(1:Nmax, level = 1:Nmax, labels = 1:Nmax), each = N)
RATIOS <- c()
for (j in 1:Nmax) {
  for (i in 1:N) {
    RATIOS <- append(RATIOS, as.numeric(data[2 * (j - 1) + 2, i]) / as.numeric(data[2 * (j - 1) + 1, i]))
  }
}
boxData <- data.frame(list(Number_of_Patients = RUN, Ratios = RATIOS))

ggplot(boxData, aes(Number_of_Patients, Ratios)) + geom_boxplot()


##########################################################################
## Friberg-Karlsson Model: vv population regime
## Pass initial states and parameters as parameters
## One set of initial estimates and parameters 
data <- read.csv(file = "mixSolverResult_pop_vv.csv")
data$X <- NULL
data$X0 <- NULL
N <- length(head(data))  ## expect 100

## summary plots
# 1: plot ratio against number of parameters (9 * number of "patients")
Nmax <- nrow(data) / 2;
data_num <- rep(0, N)
data_mix <- rep(0, N)
ratios <- rep(0, Nmax)

## FIX ME - may be a way to do this w/o for loop
for (j in 0:(Nmax - 1)) {
  for (i in 1:N) { 
    data_num[i] <- data[2 * j + 1, i]
    data_mix[i] <- data[2 * j + 2, i]
  }
  ratios[j + 1] <- mean(data_mix) / mean(data_num)
}

## very basic plot
plot(seq(1:Nmax), ratios)


# 2: plot box plots to show spread of ratios
## FIX ME - code w/o for loop
boxData <- data.frame()
RUN <- rep(factor(1:Nmax, level = 1:Nmax, labels = 1:Nmax), each = N)
RATIOS <- c()
for (j in 1:Nmax) {
  for (i in 1:N) {
    RATIOS <- append(RATIOS, as.numeric(data[2 * (j - 1) + 2, i]) / as.numeric(data[2 * (j - 1) + 1, i]))
  }
}
boxData <- data.frame(list(Number_of_Patients = RUN, Ratios = RATIOS))

ggplot(boxData, aes(Number_of_Patients, Ratios)) + geom_boxplot()


##########################################################################
## One Compartment Model with 1st Order Absorption

### dd regime
data_s_dd <- read.csv(file = "mixSolverResult.csv")
N_s_dd <- length(head(data_s_dd) - 1)
data_s_dd_num <- rep(0, N_s_dd)
data_s_dd_mix <- data_s_dd_num
for (i in 1:N_s_dd) data_s_dd_num[i] <- data_s_dd[1, i]
for (i in 1:N_s_dd) data_s_dd_mix[i] <- data_s_dd[2, i]

mean(data_s_dd_num)
# 0.661478
# simple model: 0.0004352567
mean(data_s_dd_mix)
# 0.6587093
# simple model: 0.000387018

### vv regime
data_s_vv <- read.csv(file = "mixSolverResult_vv.csv")
N_s_vv <- length(head(data_s_vv) - 1)
data_s_vv_num <- rep(0, N_s_vv)
data_s_vv_mix <- data_s_vv_num
for (i in 1:N_s_vv) data_s_vv_num[i] <- data_s_vv[1, i]
for (i in 1:N_s_vv) data_s_vv_mix[i] <- data_s_vv[2, i]

mean(data_s_vv_num)
# 0.05603206
mean(data_s_vv_mix)
# 0.0199685

ratio <-  mean(data_s_vv_mix) / mean(data_s_vv_num)
ratio
# 0.3563762

##########################################################################
## Friberg-Karlsson Model with 1st Order Absorption

## dd regime
data_dd <- read.csv(file = "mixSolverResult_dd.csv")
N_dd <- length(head(data_dd) - 1)

data_dd_num <- rep(0, N_dd)
data_dd_mix <- data_dd_num
for (i in 1:N_dd) data_dd_num[i] <- data_dd[1, i]
for (i in 1:N_dd) data_dd_mix[i] <- data_dd[2, i]

mean(data_dd_num)
# 1.83966e-05
mean(data_dd_mix)
# 2.756044e-05


## vv regime
data_vv <- read.csv(file = "mixSolverResult_vv.csv")
N_vv <- length(head(data_vv) - 1)

data_vv_num <- rep(0, N_vv)
data_vv_mix <- data_vv_num
for (i in 1:N_vv) data_vv_num[i] <- data_vv[1, i]
for (i in 1:N_vv) data_vv_mix[i] <- data_vv[2, i]

mean(data_vv_num)
# 0.0005151409
mean(data_vv_mix)
# 0.004688307
