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


class SolverADAM(SolverAbstract):
    def __init__(self, shootingProblem):
        SolverAbstract.__init__(self, shootingProblem)
        self.alphas = [10 ** (-n) for n in range(-1, 10)]
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
        self.th_stop = 1.e-9
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
        # self.Vxx[-1][:, :] = self.problem.terminalData.Lxx
        self.dJdx[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):

            self.dJdu[t, :] = data.Lu + self.dJdx[t+1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t+1, :] @ data.Fx


    def forwardPass(self, alpha, i):
        cost_try = 0.
        self.Vdu = self.Beta1 * self.Vdu + (1-self.Beta2) * self.dJdu
        self.Sdu = self.Beta2 * self.Sdu + (1-self.Beta2) * self.dJdu
        Vdu_corrected = self.Vdu / (1 - self.Beta1 ** (i+1))
        Sdu_corrected = self.Sdu / (1 - self.Beta2 ** (i+1))

        print('Sdu_corrected:', Sdu_corrected)
        update = Vdu_corrected / (np.sqrt(Sdu_corrected) + self.eps)
        us = np.array(self.us)
        us_try = us - alpha * update
        self.us_try = list(us_try)

        # need to set xs_try[0] = x0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])
            self.xs_try[t + 1] = data.xnext
            cost_try += data.cost

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])

        cost_try += self.problem.terminalData.cost

        return cost_try

    def tryStep(self, alpha, i):

        self.cost_try = self.forwardPass(alpha, i)
        print('tryStep:', self.cost_try)

        return self.cost - self.cost_try

    def solve(self, init_xs=None, init_us=None, maxIter=10000, isFeasible=True):
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

            for a in self.alphas:  # forward pass with line search
                Vdu_p = self.Vdu
                Sdu_p = self.Sdu
                try:
                    print('enter tryStep')
                    dV = self.tryStep(a, i)

                except:
                    # repeat starting from a smaller alpha
                    print("Try Step Failed for alpha = %s" % a)
                    continue

                if dV > self.threshold:
                    print("step accepted, new cost is %s" % ( self.cost_try))
                    self.setCandidate(self.xs_try, self.us_try, isFeasible)
                    self.cost = self.cost_try
                    if dV < 1.e-12:
                        self.n_little_improvement += 1
                        print("little improvements")
                    #break
                else:
                    self.Vdu = Vdu_p
                    self.Sdu = Sdu_p

            self.stoppingCriteria()
            print('stop:', self.stop)
            print('th_stop:', self.th_stop)

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
        self.Vdu = np.zeros_like(self.dJdu)
        self.Sdu = np.zeros_like(self.dJdu)
        self.Beta1 = .9
        self.Beta2 = .999
        self.eps = 1e-8
        # self.dx = [np.zeros(m.state.ndx) for m in self.models()]
        # self.Qu = [np.zeros([m.nu]) for m in self.problem.runningModels]
        # self.Qx = [np.zeros([m.state.ndx]) for m in self.problem.runningModels]
        # self.Qxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.problem.runningModels]
        # self.Qux = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        # self.Quu = [np.zeros([m.nu, m.nu]) for m in self.problem.runningModels]
        # self.Vxx = [np.zeros([m.state.ndx, m.state.ndx]) for m in self.models()]
        # self.Vx = [np.zeros(m.state.ndx) for m in self.models()]
        # self.dv = [0. for _ in self.models()]
        # self.K = [np.zeros([m.nu, m.state.ndx]) for m in self.problem.runningModels]
        # self.k = [np.zeros([m.nu]) for m in self.problem.runningModels]
