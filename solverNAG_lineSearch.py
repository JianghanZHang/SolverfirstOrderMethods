""""
This implementation is based on https://link.springer.com/article/10.1007/s00245-020-09718-8
"""
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


class SolverNAG(SolverAbstract):
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
            self.dJdu[t, :] = data.Lu + self.dJdx[t + 1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t + 1, :] @ data.Fx

        self.Qu = self.dJdu
        self.kkt = linalg.norm(self.Qu, 2)
        self.KKTs.append(self.kkt)
        # pdb.set_trace()

    def forwardPass(self, alpha, i):
        cost_try = 0.
        us = np.array(self.us)
        self.m = (self.mu * self.m) + ((1+self.mu) * self.dJdu - self.mu * self.dJdu_p)
        # m_k = MU * m_k-1 + g_k'; g_k' = (1+MU) * g_k - MU * g_k-1
        if i == 0:
            self.m = self.dJdu
        # pdb.set_trace()
        us_try = us - alpha * self.m
        self.us_try = list(us_try)

        self.curvature_0 = 0.
        # need to make sure self.xs_try[0] = x0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])
            self.xs_try[t + 1] = data.xnext
            cost_try += data.cost
            self.curvature_0 += self.dJdu[t, :].T @ self.direction[t, :]

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])

        cost_try += self.problem.terminalData.cost

        return cost_try

    def tryStep(self, alpha, i):
        self.direction_p = self.direction
        self.cost_try = self.forwardPass(alpha, i)

        return self.cost - self.cost_try

    def refresh_(self):
        self.alpha = 1.
        self.alpha_p = 1.
        self.m = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])

    def solve(self, init_xs=None, init_us=None, maxIter=100, isFeasible=False):
        # ___________________ Initialize ___________________#
        if init_xs is None:
            init_xs = [np.zeros(m.state.nx) for m in self.models()]
        if init_us is None:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels]

        init_xs[0][:] = self.problem.x0.copy()  # Initial condition guess must be x0
        self.xs_try[0][:] = self.problem.x0.copy()

        if not isFeasible:
            init_xs = self.problem.rollout(init_us)

        self.setCandidate(init_xs, init_us, False)
        # pdb.set_trace()
        #self.refresh_()

        self.cost = self.calc()  # self.forwardPass(1.)  # compute initial value for merit function
        self.costs.append(self.cost)
        # print("initial cost is %s" % self.cost)

        for i in range(maxIter):
            self.numIter = i
            self.guess = 1.#min(2 * self.alpha_p, 1)
            self.alpha = self.guess
            recalc = True  # this will recalculate derivatives in computeDirection
            while True:  # backward pass
                try:
                    self.computeDirection(recalc=recalc)

                except:
                    print('In', i, 'th iteration.')
                    # import pdb; pdb.set_trace()
                    raise BaseException("Backward Pass Failed")
                break

            if self.kkt < self.th_stop:
                print('Converged')
                return True

            while True:  # doing line search
                while True:  # forward pass
                    try:
                        self.tryStep(self.alpha, i)
                    except:
                        # repeat starting from a smaller alpha
                        print("Try Step Failed for alpha = %s" % self.alpha)
                        raise BaseException('FP failed')
                    break

                if self.cost_try <= self.cost + self.c1 * self.alpha * self.curvature_0:
                    # line search succeed -> exit
                    self.setCandidate(self.xs_try, self.us_try, True)
                    self.cost = self.cost_try
                    self.costs.append(self.cost)
                    self.alpha_p = self.alpha
                    break
                else:
                    self.alpha *= .5

                if self.alpha < 2**(-15):
                    print(f'alpha={self.alpha}, line search failed')
                    return False

            if self.alpha == self.guess:
                self.guess_accepted.append(True)
            else:
                self.guess_accepted.append(False)

            self.stoppingCriteria()

        return False

    def stoppingCriteria(self):
        if self.dV < 1e-12:
            self.n_little_improvement += 1
            if VERBOSE: print('Little improvement.')

    def allocateData(self):
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()]
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.dJdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.alpha_p = 0
        self.dJdu_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.numIter = 0
        self.costs = []
        self.kkt = 0.
        self.KKTs = []
        self.m = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.alpha = 1.
        self.curvature_0 = 0.
        self.alpha_p = 1.
        self.guess_accepted = []
        self.mu = .95
        self.y = 0.
        self.y_p = 0.
        self.us_p = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.s = 0.
        self.s_p = 0.
        self.v = 0.



