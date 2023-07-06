import numpy as np
from numpy import linalg

import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract

#############################################################
# This solver fail to converge in arm_manipulation example #
#                                                           #
#############################################################

LINE_WIDTH = 100

VERBOSE = False


def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class SolverGD(SolverAbstract):
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
        self.past_grad = 0.
        self.change_p = self.change
        self.change = 0.
        self.dJdx[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu[t, :] = data.Lu + self.dJdx[t+1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t+1, :] @ data.Fx
            self.change = self.dJdu[t, :].T @ (-self.dJdu[t, :])


    def forwardPass(self, alpha):
        cost_try = 0.
        us = np.array(self.us)
        self.direction = -self.dJdu
        us_try = us + alpha * self.direction
        self.us_try = list(us_try)
        self.lb = 0.
        self.ub = 0.
        self.curr_grad = 0.
        # need to make sure self.xs_try[0] = x0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])
            self.xs_try[t + 1] = data.xnext
            cost_try += data.cost
            self.lb += -self.c * alpha * self.dJdu[t, :].T @ self.direction[t, :]  # For Wolfe condition (sufficient decrease)
            self.ub += (self.c - 1) * alpha * self.dJdu[t, :].T @ self.direction[t, :]
            print('g.T@direction:', self.dJdu[t, :].T @ self.direction[t, :])
            #                 -c1*a*grad*direction
            # self.curr_grad = abs(data.Lu.T @ self.direction[t, :]) #grad(alpha)Tpk !!this is wrong!!
            # Computing for the current curvature information requires another backward pass,
            # which is useless if turns out that the curvature condition was not satisfied.


        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])

        cost_try += self.problem.terminalData.cost

        return cost_try

    def init_alpha(self):
        return self.alpha_p * (self.change_p/self.change)

    def tryStep(self, alpha):
        self.direction_p = self.direction
        self.cost_try = self.forwardPass(alpha)
        # print('tryStep', self.cost_try)
        # print('alpha', alpha)

        return self.cost - self.cost_try

    def solve(self, init_xs=None, init_us=None, maxIter=100, isFeasible=True):
        # ___________________ Initialize ___________________#
        if init_xs is None:
            init_xs = [np.zeros(m.state.nx) for m in self.models()]
        if init_us is None:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels]

        init_xs[0][:] = self.problem.x0.copy()  # Initial condition guess must be x0
        self.xs_try[0][:] = self.problem.x0.copy()

        self.setCandidate(init_xs, init_us, False)
        # compute the gaps

        self.cost = self.calc()  # self.forwardPass(1.)  # compute initial value for merit function

        print("initial cost is %s" % self.cost)

        for i in range(maxIter):
            recalc = True  # this will recalculate derivatives in computeDirection
            while True:  # backward pass
                try:
                    self.computeDirection(recalc=recalc)

                except:
                    print('In', i, 'th iteration.')
                    raise BaseException("Backward Pass Failed")
                break

            failed = False
            using_prev_info = False
            # initialize step length
            if i > 0:
                alpha = self.init_alpha()
                using_prev_info = True
            else:
                alpha = 1

            while True:  # forward pass with line search

                if failed and using_prev_info:
                    alpha = 1
                    failed = False
                    using_prev_info = False

                try:

                    dV = self.tryStep(alpha)

                except:
                    # repeat starting from a smaller alpha
                    print("Try Step Failed for alpha = %s" % alpha)
                    raise BaseException("Forward Pass Failed")
                    continue

                # print('dV:', dV)
                #
                # print('lower bound:', self.lb)
                # print('upper bound:', self.ub)
                #
                # print('lower bound satisfied:', dV > self.lb)
                # print('upper bound satisfied:', dV < self.ub)


                #if dV > self.threshold and self.curr_grad < self.past_grad:  # check for Wolfe condition
                if self.ub >= dV >= self.lb:  # check for the Goldstein conditon
                    print("step accepted for alpha = %s \nnew cost is %s" % (alpha, self.cost_try))
                    self.setCandidate(self.xs_try, self.us_try, isFeasible)
                    self.cost = self.cost_try
                    self.alpha_p = alpha
                    if dV < 1e-12:
                        self.n_little_improvement += 1
                        print("little improvements")
                    break

                if alpha <= 2**(-15):
                    print("No decrease found")
                    return False

                alpha /= 2
                failed = True
                #print('step length failed at', i, 'iteration.')

            self.stoppingCriteria()

            if self.n_little_improvement >= 1 or self.stop < self.th_stop:
                print('Converged')
                return True

        return False

    def stoppingCriteria(self):
        self.stop = 0
        T = self.problem.T
        for t in range(T):
            self.stop += linalg.norm(self.dJdu[t])

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

