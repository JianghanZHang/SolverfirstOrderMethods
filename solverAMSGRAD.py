#https://towardsdatascience.com/complete-guide-to-adam-optimization-1e5f29532c3d
import pdb

import numpy as np
from numpy import linalg

import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract

LINE_WIDTH = 100

VERBOSE = False


def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class SolverAMSGRAD(SolverAbstract):
    def __init__(self, shootingProblem):
        SolverAbstract.__init__(self, shootingProblem)
        self.cost = 0.
        self.cost_try = 0.
        self.threshold = 1e-12
        self.stop = 0.
        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.th_step = .5
        self.th_stop = 1.e-5
        self.n_little_improvement = 0
        self.c1 = 1e-4
        self.c2 = .9
        self.c = 1e-4
        self.past_grad = 0.
        self.curr_grad = 0.
        self.change = 0.
        self.change_p = 0.
        self.lb = 0.
        self.ub = 0.
        self.allocateData()

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod

    def calc(self):
        # compute cost and derivatives at deterministic nonlinear trajectory

        self.problem.calc(self.xs, self.us)
        cost = self.problem.calcDiff(self.xs, self.us)
        return cost

    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.dJdu_p = self.dJdu
        self.backwardPass()

    def backwardPass(self):
        self.dJdx[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu[t, :] = data.Lu + self.dJdx[t+1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t+1, :] @ data.Fx

        self.Qu = self.dJdu
        self.kkt = linalg.norm(self.Qu, 2)
        self.KKTs.append(self.kkt)

    def forwardPass(self, alpha, i):
        cost_try = 0.
        self.v_max_p = self.v_max
        self.m = self.Beta1 * self.m + (1 - self.Beta1) * self.dJdu
        self.v = self.Beta2 * self.v + (1 - self.Beta2) * (self.dJdu**2)
        self.v_max = np.maximum(self.v_max_p, self.v)
        if self.bias_correction:
            m_corrected = self.m / (1 - self.Beta1 ** (i + 2))
            v_corrected = self.v_max / (1 - self.Beta2 ** (i + 2))
        else:
            m_corrected = self.m
            v_corrected = self.v_max
        update = m_corrected / (np.sqrt(v_corrected) + self.eps)
        us = np.array(self.us)
        us_try = us - alpha * update
        self.us_try = list(us_try)

        # need to make sure self.xs_try[0] = x0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])
            self.xs_try[t + 1] = data.xnext
            cost_try += data.cost

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])

        cost_try += self.problem.terminalData.cost

        return cost_try

    def tryStep(self, alpha, i):
        self.cost_try = self.forwardPass(alpha, i)
        return self.cost - self.cost_try

    def solve(self, init_xs=None, init_us=None, maxIter=100, isFeasible=False, alpha=None):
        # ___________________ Initialize ___________________#
        if init_xs is None:
            init_xs = [np.zeros(m.state.nx) for m in self.models()]
        if init_us is None:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels]

        init_xs[0][:] = self.problem.x0.copy()  # Initial condition guess must be x0
        self.xs_try[0][:] = self.problem.x0.copy()

        if not isFeasible:
            init_xs = self.problem.rollout(init_us)

        if self.refresh:
            self.refresh()
        else:
            m = list(self.m[1:]) + [self.m[-1]]
            v = list(self.v[1:]) + [self.v[-1]]
            v_max = list(self.v_max[1:]) + [self.v_max[-1]]
            self.m = np.array(m)
            self.v = np.array(v)
            self.v_max = np.array(v_max)
            self.v = (self.decay2) * self.v
            self.v_max = (self.decay2) * self.v

        self.setCandidate(init_xs, init_us, False)

        self.cost = self.calc()  # self.forwardPass(1.)  # compute initial value for merit function
        self.costs.append(self.cost)

        #print("initial cost is %s" % self.cost)

        for i in range(maxIter):
            self.numIter = i
            recalc = True  # this will recalculate derivatives in computeDirection
            while True:  # backward pass
                try:
                    self.computeDirection(recalc=recalc)

                except:
                    print('In', i, 'th iteration.')
                    raise BaseException("Backward Pass Failed")
                break


            while True:  # forward pass with line search
                try:
                    self.tryStep(alpha, i)

                except:
                    # repeat starting from a smaller alpha
                    print("Try Step Failed for alpha = %s" % alpha)
                    raise BaseException("Backward Pass Failed")
                break

            self.dV = self.cost - self.cost_try
            self.setCandidate(self.xs_try, self.us_try, isFeasible)
            self.cost = self.cost_try
            self.costs.append(self.cost)

            self.stoppingCriteria()

            if self.kkt < self.th_stop:
                print('Converged')
                return True

        return False

    def stoppingCriteria(self):
        if self.dV < 1e-12:
            self.n_little_improvement += 1
            if VERBOSE: print('Little improvement.')

    def refresh_(self):

        self.dJdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.m = np.zeros_like(self.dJdu)
        self.v = np.zeros_like(self.dJdu)
        self.v_max = np.zeros_like(self.dJdu)
        self.v_max_p = np.zeros_like(self.dJdu)
        self.kkt = 0.

    def allocateData(self):

        self.xs_try = [np.zeros(m.state.nx) for m in self.models()]
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.dJdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.m = np.zeros_like(self.dJdu)
        self.v = np.zeros_like(self.dJdu)
        self.Beta1 = .9
        self.Beta2 = .999
        self.eps = 1e-8
        self.kkt = 0.
        self.KKTs = []
        self.costs = []
        self.numIter = 0
        self.bias_correction = True
        self.v_max = np.zeros_like(self.dJdu)
        self.v_max_p = np.zeros_like(self.dJdu)
        self.decay1 = 1.
        self.decay2 = 1.
        self.refresh = False


