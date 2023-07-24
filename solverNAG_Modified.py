"""
http://www.damtp.cam.ac.uk/user/hf323/M19-OPT/lecture5.pdf
"""
import numpy as np
from numpy import linalg
import pdb
import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract
import time

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
        self.alphas = [2 ** (-n) for n in range(10)]
        self.cost = 0.
        self.cost_try = 0.
        self.threshold = 0.
        self.stop = 0.
        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.th_step = .5
        self.th_stop = 1e-5
        self.eps = .01
        self.mu = .8
        self.n_little_improvement = 0
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
        self.backwardPass()

    def backwardPass(self):
        self.dJdx[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu[t, :] = data.Lu + self.dJdx[t + 1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t + 1, :] @ data.Fx

        self.kkt = np.max(abs(self.dJdu))
        if VERBOSE: print(f'KKT = {self.kkt}')
        self.KKTs.append(self.kkt)

    def forwardPass(self, us_try):
        cost_try = 0.
        self.us_try = list(us_try)
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])
            # model.calcDiff(data, self.xs_try[t], self.us_try[t])
            self.xs_try[t + 1] = data.xnext
            cost_try += data.cost

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])
        cost_try += self.problem.terminalData.cost

        return cost_try

    def calcGradientNorm(self):
        # This function assume that forwardPass has been run
        self.problem.calc(self.xs_try, self.us_try)
        self.problem.calcDiff(self.xs_try, self.us_try)

        self.dJdx_try[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu_try[t, :] = data.Lu + data.Fu.T @ self.dJdx_try[t + 1, :]
            self.dJdx_try[t, :] = data.Lx + data.Fx.T @ self.dJdx_try[t + 1, :]

        gradientNorm = linalg.norm(self.dJdu_try, ord=2)
        return gradientNorm

    def tryStep(self, alpha):
        us = np.array(self.us)  # x_k
        # tmp = (self.theta_p ** 2) * (alpha / self.alpha_p)  # a^2 = theta_k-1^2 * (t_k/t_k-1)
        # self.theta = .5 * (-np.sqrt(tmp) + np.sqrt(tmp + 4))  # theta_k = (-a+sqrt(a^2+4))/2
        self.theta = (np.sqrt(alpha) * self.theta_p * np.sqrt(4*self.alpha_p+alpha * (self.theta_p**2))
                       -(alpha * (self.theta_p**2))) / (2 * self.alpha_p)

        assert abs(((1-self.theta) * alpha) / self.theta**2 - self.alpha_p / self.theta_p**2) < 1e-6
        self.y = (1 - self.theta) * us + self.theta * self.v  # y = (1 - theta_k) * x_k + theta_k * v_k
        cost_y = self.forwardPass(self.y)  # compute f(y)
        gradient_norm_y = self.calcGradientNorm()  # compute ||grad(f(y))||
        self.threshold = -.5 * self.alpha * (gradient_norm_y ** 2)
        us_try = self.y - alpha * self.dJdu_try  # x_k+1 = y - t_k * grad(f(y))
        self.v = us + (1/self.theta) * (us_try - us)
        self.cost_try = self.forwardPass(us_try)  # f(x_k+1)

        return self.cost_try - cost_y

    def solve(self, init_xs=None, init_us=None, maxIter=100, isFeasible=True):
        # ___________________ Initialize ___________________#
        if init_xs is None:
            init_xs = [np.zeros(m.state.nx) for m in self.models()]
        if init_us is None:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels]
        if not isFeasible:
            init_xs = self.problem.rollout(init_us)

        init_xs[0][:] = self.problem.x0.copy()  # Initial condition guess must be x0
        self.xs_try[0][:] = self.problem.x0.copy()

        self.setCandidate(init_xs, init_us, True)
        # compute the gaps

        self.cost = self.calc()  # self.forwardPass(1.)  # compute initial value for merit function

        self.costs.append(self.cost)

        print("initial cost is %s" % self.cost)
        self.numIter = 0
        self.guess_accepted = []

        self.v = np.array(self.us)  # v_0 = x_0

        for i in range(maxIter):

            # if self.kkt < self.th_stop:
            #     print('Converged')
            #     return True
            self.alpha = 2.
            for k in range(15):

                while True:
                    try:
                        dV = self.tryStep(self.alpha)
                    except:
                        # repeat starting from a smaller alpha
                        raise BaseException(f"Try Step Failed in iteration {i}.")
                    break

                if dV <= self.threshold:
                    print("step accepted for alpha = %s \n new cost is %s" % (self.alpha, self.cost_try))
                    self.setCandidate(self.xs_try, self.us_try, isFeasible)
                    self.cost = self.cost_try
                    self.costs.append(self.cost)
                    self.alpha_p = self.alpha
                    self.theta_p = self.theta
                    break
                else:
                    self.alpha *= .5
                    print(f'alpha= {self.alpha}')

                if self.alpha < 2 ** (-10):
                    print(f'line search failed')
                    return False

            self.setCandidate(self.xs_try, self.us_try, isFeasible)
            self.cost = self.cost_try
            self.costs.append(self.cost)
            self.numIter += 1

        return False

    def stoppingCriteria(self):
        pass
        # self.stop = 0
        # T = self.problem.T
        # for t in range(T):
        #     self.stop += linalg.norm(self.Qu[t])

    def allocateData(self):
        self.y = 0.
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()]
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.dJdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdu_try = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdu_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.q = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.dJdx_try = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.alpha = 0.
        self.alpha_p = 2.
        self.theta_p = 1.
        self.cost_p = 0.
        self.v = 0.
        self.costs = []
        self.alphas = []
        self.gammas = []
        self.KKTs = []
        self.kkt = 0.