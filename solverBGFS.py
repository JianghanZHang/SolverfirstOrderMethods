import numpy as np
import scipy.linalg as scl
from crocoddyl import SolverAbstract
from numpy import linalg

LINE_WIDTH = 100

VERBOSE = False


def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class SolverBGFS(SolverAbstract):
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
        self.th_stop = 1.e-9
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
        self.memory_length = 30
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

    def computeDirection(self, num_iter, recalc=True):

        self.direction_p = self.direction.copy()
        self.dJdu_p = self.dJdu.copy()
        # if recalc:
        #    if VERBOSE: print("Going into Calc from compute direction")

        self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.backwardPass(num_iter)  # get new dJdu
        self.q = self.dJdu.copy()

        for i in range(min(self.memory_length, num_iter)-1, -1, -1):
            for j in range(self.problem.T):
                # print('self.y[i][j][:].T @ self.s[i][j][:]:', self.y[i][j, :].T @ self.s[i][j, :])

                # rho should be a scalar
                self.rho[i, j] = 1 / (self.y[i][j, :].T @ self.s[i][j, :])

                # print('rho:', self.rho)
                # print('self.s[i][j][:].T @ self.q[j][:]:', self.s[i][j, :].T @ self.q[j][:])

                # aux0[i, j] should be a scalar
                self.aux0[i, j] = self.rho[i, j] * (self.s[i][j, :].T @ self.q[j, :])

                # print('aux0:', self.aux0)
                # print('aux0[i, j]', self.aux0[i, j])
                # print('q:', self.q)

                self.q[j, :] -= self.aux0[i, j] * self.y[i][j, :]
                # print('q:', self.q)

        H_init = self.init_hessian_approx(num_iter)  # Previous y is y[-2], previous s is s[-1]

        for j in range(self.problem.T):
            self.r[j] = H_init[j][:, :] @ self.q[j, :]

        for i in range(0, min(self.memory_length, num_iter), 1):
            for j in range(self.problem.T):
                self.aux1[j] = self.rho[i, j] * self.y[i][j, :].T @ self.r[j][:]  # aux1 should be a scalar
                self.r[j] += (self.aux0[i, j] - self.aux1[j]) * self.s[i][j, :]

        self.direction = -self.r
        # print('direction:', self.direction)

    def backwardPass(self, num_iter):
        self.dJdx[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu[t, :] = data.Lu + self.dJdx[t + 1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t + 1, :] @ data.Fx

        if num_iter - 1 < self.memory_length and num_iter != 0:
            self.y.append(self.dJdu - self.dJdu_p)  # y keeps track of the most recent m steps
        else:
            self.y.append(self.dJdu - self.dJdu_p)
            self.y.pop(0)

    def init_hessian_approx(self, num_iter):
        H_init = self.H0[:]

        if True:  # num_iter <= 1:
            return H_init

        else:
            for t in range(self.problem.T):
                num = self.s[-1][t, :].T @ self.y[-1][t, :]  # should be a scalar
                den = self.y[-1][t, :].T @ self.y[-1][t, :]  # should be a scalar
                # print('den:', den)
                # print('num:', num)
                H_init[t][:, :] = (num / den) * self.H0[t][:, :]
            return H_init

    def forwardPass(self, alpha):
        cost_try = 0.
        us = np.array(self.us)
        us_try = us + alpha * self.direction
        self.us_try = list(us_try)
        self.lb = 0.
        self.ub = 0.
        self.threshold = 0.
        self.curr_grad = 0.
        # need to make sure self.xs_try[0] = x0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])
            self.xs_try[t + 1] = data.xnext
            cost_try += data.cost
            self.lb += -self.c * alpha * self.dJdu[t, :].T @ self.direction[t, :]
            self.ub += (self.c - 1) * alpha * self.dJdu[t, :].T @ self.direction[t, :]

            # For Wolfe condition (sufficient decrease)
            self.threshold += -self.c1 * alpha * self.dJdu[t, :].T @ self.direction[t, :]

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])

        cost_try += self.problem.terminalData.cost

        return cost_try

    def init_alpha(self):
        return 1  # self.alpha_p * (self.change_p / self.change)

    def calcCurvature(self):
        # For Wolfe condition (curvature condition)
        curvature = 0.
        self.dJdx_try[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu[t, :] = data.Lu + self.dJdx[t + 1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t + 1, :] @ data.Fx
            curvature += self.dJdu[t, :].T @ self.direction[t, :]

        return curvature

    def tryStep(self, alpha):
        self.curvature_curr = 0.
        self.curvature_prev = 0.

        # self.curvature_prev = self.c2 * self.calcCurvature()
        self.cost_try = self.forwardPass(alpha)
        # self.problem.calc(self.xs_try, self.us_try)
        # self.problem.calcDiff(self.xs_try, self.us_try)
        # self.curvature_curr = self.calcCurvature()

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
                    self.computeDirection(i, recalc=True)

                except:
                    print('In', i, 'th iteration.')
                    raise BaseException("Backward Pass Failed")
                break

            alpha = 1

            while True:  # forward pass with line search

                try:
                    dV = self.tryStep(alpha)

                except:
                    # repeat starting from a smaller alpha
                    print("Try Step Failed for alpha = %s" % alpha)
                    raise BaseException("Forward Pass Failed")
                    continue
                # print('dV:', dV)
                # print('threshold:', self.threshold)
                # print('curvature_prev:', self.curvature_prev)
                # print('curvature_curr:', self.curvature_curr)

                if dV >= max(self.threshold, 0):
                    # print(f'in {i}th iteration:')
                    # print("step accepted for alpha = %s \n new cost is %s" % (alpha, self.cost_try))
                    if i < self.memory_length:  # keep track of the most recent m steps
                        self.s.append(alpha * self.direction)
                    else:
                        self.s.append(alpha * self.direction)
                        self.s.pop(0)

                    self.setCandidate(self.xs_try, self.us_try, isFeasible)
                    self.cost = self.cost_try
                    if dV < 1.e-12:
                        self.n_little_improvement += 1
                        print("little improvements")

                    break

                else:
                    self.calc()

                if alpha <= 1e-10:
                    print("No decrease found")
                    return False

                alpha *= .5
                # print('step length failed at', i, 'iteration.')
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
        self.dJdu_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.q = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.alpha_p = 0
        self.direction = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        # self.y = [np.array([np.zeros(m.nu) for m in self.problem.runningModels]) for n in range(self.memory_length)]
        # self.s = [np.array([np.zeros(m.nu) for m in self.problem.runningModels]) for n in range(self.memory_length)]
        self.y = []
        self.s = []
        self.H0 = [np.eye(m.nu) for m in self.problem.runningModels]
        self.r = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.rho = np.zeros([self.memory_length, self.problem.T])
        self.aux0 = np.zeros([self.memory_length, self.problem.T])
        self.aux1 = np.zeros([self.problem.T])
        self.curvature_curr = 0.
        self.curvature_prev = 0.
