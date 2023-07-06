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


class SolverLBGFS(SolverAbstract):
    def __init__(self, shootingProblem):
        SolverAbstract.__init__(self, shootingProblem)
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
        self.th_stop = 1.e-5
        self.n_little_improvement = 0
        self.c1 = 1e-4
        self.c2 = 1.
        #self.c = 1e-4
        self.past_grad = 0.
        self.curr_grad = 0.
        self.change = 0.
        self.change_p = 0.
        self.lb = 0.
        self.ub = 0.
        self.memory_length = 1000
        self.alpha_threshold = 1e-10
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

        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")

        self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.backwardPass(num_iter) # get new dJdu
        self.q = self.dJdu.copy()

        for i in range(min(self.memory_length, num_iter) - 1, -1, -1):
            tmp = 0
            for j in range(self.problem.T):
                # rho should be a scalar
                tmp += (self.y[i][j, :].T @ self.s[i][j, :])
            self.rho[i] = 1 / tmp 

            self.aux0[i] = 0
            for j in range(self.problem.T):
                # aux0[i, j] should be a scalar
                self.aux0[i] += self.rho[i] * (self.s[i][j, :].T @ self.q[j, :])
            self.q -= self.aux0[i] * self.y[i]

        H_init = self.init_hessian_approx(num_iter)  # Previous y is y[-2], previous s is s[-1]

        for j in range(self.problem.T):
            self.r[j] = H_init[j][:, :] @ self.q[j, :]

        self.aux1 = 0
        for i in range(0, min(self.memory_length, num_iter), 1):
            for j in range(self.problem.T):
                self.aux1 += self.rho[i] * self.y[i][j, :].T @ self.r[j][:]  # aux1 should be a scalar
            for j in range(self.problem.T):
            # import pdb; pdb.set_trace()
                self.r[j] += (self.aux0[i] - self.aux1) * self.s[i][j, :]

        self.direction = -self.r

    def backwardPass(self, num_iter):

        self.dJdx[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu[t, :] = data.Lu + data.Fu.T @ self.dJdx[t + 1, :]
            self.dJdx[t, :] = data.Lx + data.Fx.T @ self.dJdx[t + 1, :]
        if num_iter - 1 < self.memory_length and num_iter != 0:
            self.y.append(self.dJdu - self.dJdu_p)  # y keeps track of the most recent m steps
        else:
            self.y.append(self.dJdu - self.dJdu_p)
            self.y.pop(0)


    def init_hessian_approx(self, num_iter):
        H_init = self.H0.copy()

        return H_init
        if num_iter == 0:
            return H_init

        else:
            for t in range(self.problem.T):
                num = self.s[-1][t, :].T @ self.y[-1][t, :]  # should be a scalar
                den = self.y[-1][t, :].T @ self.y[-1][t, :]  # should be a scalar
                # print('den:', den)
                # print('num:', num)
                H_init[t][:, :] = (num / den) * self.H0[t][:, :].copy()
            return H_init

    def calcCurvature(self):
        curvature = 0.
        # For Wolfe condition (curvature condition)
        self.dJdx_try[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu_try[t, :] = data.Lu + data.Fu.T @ self.dJdx_try[t + 1, :]
            self.dJdx_try[t, :] = data.Lx + data.Fx.T @ self.dJdx_try[t + 1, :]
            curvature += self.dJdu_try[t, :].T @ self.direction[t, :]

        return curvature

    def calcCurvature_abs(self):
        curvature_abs = 0.
        # For Wolfe condition (curvature condition)
        self.dJdx_try[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu_try[t, :] = data.Lu + data.Fu.T @ self.dJdx_try[t + 1, :]
            self.dJdx_try[t, :] = data.Lx + data.Fx.T @ self.dJdx_try[t + 1, :]
            curvature_abs += abs(self.dJdu_try[t, :].T @ self.direction[t, :])
            print(f'in {t}th step: grad = {self.dJdu_try[t, :].T}, direction = {self.direction[t, :]}, gradTdirection= {self.dJdu[t, :].T @ self.direction[t, :]}')

        return curvature_abs

    def forwardPass(self, alphas):
        cost_try = 0.
        us = np.array(self.us)
        us_try = us + alphas * self.direction
        self.us_try = list(us_try)
        self.lb = 0.
        self.ub = 0.
        self.threshold = 0.
        self.curr_grad = 0.

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])
            model.calcDiff(data, self.xs_try[t], self.us_try[t])
            self.xs_try[t + 1] = data.xnext
            cost_try += data.cost

            # For Wolfe condition (sufficient decrease)
            self.threshold -= (self.c1 * self.dJdu[t, :].T @ (alphas * self.direction[t, :]))

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])

        cost_try += self.problem.terminalData.cost

        return cost_try

    def tryStep(self, alphas):
        self.curvature_curr = 0.
        self.curvature_prev = 0.
        self.curvature_prev = self.calcCurvature_abs()
        print('step:', alphas * self.direction)
        self.cost_try = self.forwardPass(alphas)
        self.problem.calc(self.xs_try, self.us_try)
        self.problem.calcDiff(self.xs_try, self.us_try)
        self.curvature_curr = self.calcCurvature_abs()
        print('curvature_prev:', self.curvature_prev)
        print('curvature_curr:', self.curvature_curr)



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
                    self.computeDirection(i, recalc=True)

                except:
                    print('In', i, 'th iteration.')
                    raise BaseException("Backward Pass Failed")
                break

            alphas = 1

            while True:  # forward pass with line search

                try:
                    print(f'######################## Going into tryStep @ iteration {i} ##########################')
                    print('alpha:', alphas)
                    dV = self.tryStep(alphas)
                except:
                    # repeat starting from a smaller alpha
                    print("Try Step Failed for alpha = %s" % alphas)
                    raise BaseException("Forward Pass Failed")
                    continue

                print('dV:', dV)
                curvature_cond_satisfied = self.curvature_curr <= self.curvature_prev

                if dV >= max(self.threshold, 0) and curvature_cond_satisfied:
                    print(f'in {i}th iteration:')
                    print("step accepted for alpha = %s \n new cost is %s" % (alphas, self.cost_try))
                    if i < self.memory_length:  # keep track of the most recent m steps
                        self.s.append(alphas * self.direction)
                    else:
                        self.s.append(alphas * self.direction)
                        self.s.pop(0)

                    self.setCandidate(self.xs_try, self.us_try, isFeasible)
                    self.cost = self.cost_try
                    self.alphas_p = alphas
                    if dV < 1.e-12:
                        self.n_little_improvement += 1
                        print("little improvements")

                    break

                update_mask = .5

                self.calc()  # recalc

                alphas *= update_mask
                # print('alphas after update:', alphas)
                print('step length failed.')


                if alphas <= self.alpha_threshold:
                    print("No decrease found")
                    return False

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
        self.dJdu_try = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdu_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.q = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.dJdx_try = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.alpha_p = 0
        self.direction = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.y = []
        self.s = []
        self.H0 = [np.eye(m.nu) for m in self.problem.runningModels]
        self.r = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.rho = np.zeros(self.memory_length)
        self.aux0 = np.zeros(self.memory_length)
        self.aux1 = 0
        self.curvature = 0
        self.curvature_curr = 0.
        self.curvature_prev = 0.
        self.alphas = np.ones([self.problem.T, 1])

    def numDiff_grad(self, epsilon=1e-10):
        # initialize states and controls
        init_xs = [np.zeros(m.state.nx) for m in self.models()]
        init_us = [np.zeros(m.nu) for m in self.problem.runningModels]
        init_xs[0][:] = self.problem.x0.copy()  # Initial condition guess must be x0
        X = self.problem.rollout(init_us)
        print('X0:', X[0])
        self.setCandidate(X, init_us, True)
        # initialize completed
        horizon = len(self.us)
        print(f'horizon: {horizon}')
        cost_grad = np.zeros(horizon)
        self.calc()
        cost_minus = self.calc()
        for i in range(horizon):
            U_plus = self.us.copy()
            U = self.us.copy()
            U_plus[i] += epsilon
            # Compute the cost at U_plus and U_minus
            X_plus = self.problem.rollout(U_plus)
            print('X_plus:', X_plus[i])
            self.problem.calc(X_plus, U_plus)  # Define cost_function accordingly
            cost_plus = self.problem.calcDiff(X_plus, U_plus)
            self.calc()

            # Compute the gradient for the current element of U
            cost_grad[i] = (cost_plus - cost_minus) / epsilon

        print('numDiff_grad:', cost_grad)

        return cost_grad



