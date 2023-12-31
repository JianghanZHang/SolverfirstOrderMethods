import numpy as np
from numpy import linalg

import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract
import pdb

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
        self.cost_try_p = 0.
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
        self.c2 = .9
        #self.c = 1e-4
        self.past_grad = 0.
        self.curr_grad = 0.
        self.change = 0.
        self.change_p = 0.
        self.lb = 0.
        self.ub = 0.
        self.memory_length = 100
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

        #for j in range(self.problem.T):
        #    self.r[j] = H_init[j][:, :] @ self.q[j, :]
        r = 1 * self.q

        for i in range(0, min(self.memory_length, num_iter), 1):
            aux1 = 0
            for j in range(self.problem.T):
                aux1 += self.rho[i] * self.y[i][j, :].T @ r[j][:]  # aux1 should be a scalar
            #for j in range(self.problem.T):
            # import pdb; pdb.set_trace()
            r += (self.aux0[i]-aux1) * self.s[i]

        self.direction = -r

        print(f'In computeDirection: direction: {self.direction}')



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

    # This function is assume forwardPass has run with corresponding alpha
    # (xs_try and us_try have been updated with us_try = us + alpha * direction).
    def calcCurvature(self):
        self.problem.calc(self.xs_try, self.us_try)
        self.problem.calcDiff(self.xs_try, self.us_try)
        curvature = 0.
        # For Wolfe condition (curvature condition)
        self.dJdx_try[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu_try[t, :] = data.Lu + data.Fu.T @ self.dJdx_try[t + 1, :]
            self.dJdx_try[t, :] = data.Lx + data.Fx.T @ self.dJdx_try[t + 1, :]
            curvature += self.dJdu_try[t, :].T @ self.direction[t, :]
        return curvature

    def calcCurvature_abs(self):
        self.problem.calc(self.xs_try, self.us_try)
        self.problem.calcDiff(self.xs_try, self.us_try)
        curvature_abs = 0.
        # For Wolfe condition (curvature condition)
        self.dJdx_try[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu_try[t, :] = data.Lu + data.Fu.T @ self.dJdx_try[t + 1, :]
            self.dJdx_try[t, :] = data.Lx + data.Fx.T @ self.dJdx_try[t + 1, :]
            curvature_abs += abs(self.dJdu_try[t, :].T @ self.direction[t, :])
            print(f'in {t}th step: grad = {self.dJdu_try[t, :].T}, direction = {self.direction[t, :]}, gradTdirection= {self.dJdu[t, :].T @ self.direction[t, :]}')
        return curvature_abs

    # This function also compute the curvature_0 (when alpha = 0) and the threshold for sufficient decrease condition.
    def forwardPass(self, alpha):
        cost_try = 0.
        us = np.array(self.us)
        us_try = us + alpha * self.direction
        self.us_try = list(us_try)
        self.lb = 0.
        self.ub = 0.
        self.threshold = 0.
        self.curr_grad = 0.
        self.curvature_0 = 0.

        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            model.calc(data, self.xs_try[t], self.us_try[t])
            #model.calcDiff(data, self.xs_try[t], self.us_try[t])
            self.xs_try[t + 1] = data.xnext
            cost_try += data.cost

            # For Wolfe condition (sufficient decrease)
            self.threshold += (self.c1 * self.dJdu[t, :].T @ (alpha * self.direction[t, :]))
            self.curvature_0 += self.dJdu[t, :].T @ self.direction[t, :]

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])

        assert self.curvature_0 <= 0

        cost_try += self.problem.terminalData.cost

        return cost_try

    def tryStep(self):
        # self.curvature_curr = 0.
        # self.curvature_prev = 0.
        # self.curvature_prev = self.calcCurvature_abs()
        # self.cost = self.cost_try
        satisfied = self.lineSearch()

        return satisfied #self.cost - self.cost_try

    def lineSearch(self):
        # Throughout the line search, alpha should be monotonically increasing
        # self.cost is cost(alpha = 0)
        # Note: curvatures should be scalar
        self.alpha_max = 1.
        self.alpha =0.
        self.alpha_p = 0.

        print(f'Going into forwardPass from lineSearch initialization using alpha = max_alpha.')
        cost_try_max = self.forwardPass(self.alpha_max)
        curvature_max = self.calcCurvature()

        print(f'Going into forwardPass from lineSearch initialization using alpha = current_alpha.')
        self.cost_try = self.forwardPass(self.alpha)
        self.curvature_curr = self.calcCurvature()

        # Computing costs and curvatures for alpha_1 and alpha_0 before going into the loop
        self.cost_try_p = self.cost
        self.curvature_prev = self.curvature_0

        for i in range(0, 30):

            if self.cost_try > self.cost + self.c1 * self.alpha * self.curvature_0 or (self.cost_try >= self.cost_try_p and i > 0):
                # sufficient decrease was not satisfied
                print(f'going into zoom because sufficient decrease was not satisfied.')
                return self.zoom(self.alpha_p, self.alpha, reversed=False)

            if self.curvature_curr >= self.c2 * self.curvature_0:  # curvature(alpha = 0) is from forwardPass
                print('line search succeed.')
                # current alpha satisfy Wolfe condition -> stop line search
                return True

            if self.curvature_curr >= 0:  # alphas_i overshoot -> going into reverse zoom
                print(f'in iteration {i}, going into zoom because current curvature is positive.')
                return self.zoom(self.alpha, self.alpha_p, reverse=True)

            self.alpha_p = self.alpha
            #self.alpha *= 2
            self.alpha = self.cubicInterpolation(self.alpha, self.alpha_max, self.curvature_curr, curvature_max, self.cost_try, cost_try_max)

            self.cost_try_p = self.cost_try  # record cost(alpha_i-1) before computing new cost
            self.cost_try = self.forwardPass(self.alpha)
            # Computing new cost(alpha_i), now us_try=us + alpha_i*direction, and (xs_try, us_try) is feasible.

            self.curvature_prev = self.curvature_curr  # record curvature(alpha_i-1)
            self.curvature_curr = self.calcCurvature()
            # Computing new curvature(alpha_i), the curvature was computed using us_try=us + alpha_i*direction.
            print(f'in iteration {i}:')
            print(f'current_alpha: {self.alpha}, direction:{self.direction}')
            print(f'cost_try: {self.cost_try}; curvature(current_alpha): {self.curvature_curr}')
            print(f'cost: {self.cost}; curvature(0): {self.curvature_0}')
            print(f'cost_try_previous: {self.cost_try_p}')

        return False

    def zoom(self, alpha_lo, alpha_hi, reversed):

        # initialize the algorithm
        if reversed:  # This is the case: zoom(alpha_i, alpha_i-1)
            curvature_lo = self.curvature_curr
            curvature_hi = self.curvature_prev
            cost_try_lo = self.cost_try
            cost_try_hi = self.cost_try_p
        else:  # This is the case: zoom(alpha_i-1, alpha_i)
            curvature_lo = self.curvature_prev
            curvature_hi = self.curvature_curr
            cost_try_lo = self.cost_try_p
            cost_try_hi = self.cost_try

        for i in range(0, 30):
            self.alpha = self.cubicInterpolation(alpha_lo, alpha_hi, curvature_lo, curvature_hi, cost_try_lo, cost_try_hi)
            print(f'in zoom, current interpolated alpha: {self.alpha}. With [alpha_lo, alpha_hi] = [{alpha_lo}, {alpha_hi}]')

            self.cost_try = self.forwardPass(self.alpha)  # now us_try = us + alpha * direction
            self.curvature_curr = self.calcCurvature()
            print(f'in iteration {i}:')
            print(f'current_alpha: {self.alpha}')
            print(f'cost_try: {self.cost_try}; curvature(current_alpha): {self.curvature_curr}')
            print(f'cost: {self.cost}; curvature(0): {self.curvature_0}')
            #print(f'direction: {self.direction},\n grad: {self.dJdu}')

            if self.cost_try > self.cost + self.c1 * self.alpha * self.curvature_0 or self.cost_try >= cost_try_lo:
                print(f'sufficient decrease condition was not satisfied in zoom, changing upper bound of alpha to current alpha.')
                alpha_hi = self.alpha
                curvature_hi = self.curvature_curr
                cost_try_hi = self.cost_try

            else:

                if self.curvature_curr >= self.c2 * self.curvature_0:
                    # current alpha satisfy the Wolfe condition -> stop line search
                    print(f'line search -> zoom succeed.')
                    return True

                if self.curvature_curr * (alpha_hi - alpha_lo) >= 0:
                    print(f'reversing interval of alpha')
                    alpha_hi = alpha_lo
                    curvature_hi = curvature_lo
                    cost_try_hi = cost_try_lo

                alpha_lo = self.alpha
                curvature_lo = self.curvature_curr
                cost_try_lo = self.cost_try
                print(f'sufficient decrease condition was satisfied, '
                      f'but the curvature condition was not -> changing lower bound of alpha to current alpha.')

                # cost_try_lo = self.forwardPass(alpha_lo)  # now us_try = us + alpha_lo * direction
                # curvature_lo = self.calcCurvature()
        return False

    def cubicInterpolation(self, alpha_l, alpha_r, curvature_l, curvature_r, cost_try_l, cost_try_r):
        # Note: it is possible to have alpha_i < alpha_i-1

        return alpha_l + .5 * (alpha_r - alpha_l)

        # print('in cubicInterpolation:')
        #
        # d1 = curvature_l + curvature_r - 3 * (cost_try_l - cost_try_r) / (alpha_l - alpha_r)
        #
        # d2 = np.sign(alpha_r - alpha_l) * (d1**2 - curvature_l * curvature_r) ** .5
        #
        # print(f'd1: {d1}, d2: {d2}')
        # print(f'alpha_l: {alpha_l}, alpha_r: {alpha_r}, curvature_l: {curvature_l}, curvature_r: {curvature_r}, cost_try_l: {cost_try_l}, cost_try_r: {cost_try_r}')
        # alpha = alpha_r - (alpha_r - alpha_l) * ((curvature_r + d2 - d1) / (curvature_r - curvature_l + 2 * d2))
        #
        # # if alpha - alpha_l < 1e-6 or alpha_r - alpha < 1e-6:
        # #     alpha = alpha_r / 2
        # return alpha

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

            while True:  # Going into line search
                try:
                    print(f'######################## Going into tryStep @ iteration {i} ##########################')
                    satisfied = self.tryStep()
                except:
                    # repeat starting from a smaller alpha
                    print("Try Step Failed for alpha = %s" % self.alpha)
                    raise BaseException("Forward Pass Failed")
                    continue

                break

            print('Satisfied:', satisfied)

            if satisfied:
                print(f'in {i}th iteration:')
                print("step accepted for alpha = %s \n new cost is %s" % (self.alpha, self.cost_try))
                if i < self.memory_length:  # keep track of the most recent m steps
                    self.s.append(self.alpha * self.direction)
                else:
                    self.s.append(self.alpha * self.direction)
                    self.s.pop(0)

                self.dV = self.cost - self.cost_try
                self.setCandidate(self.xs_try, self.us_try, isFeasible)
                self.cost = self.cost_try

            else:
                print('line search failed')
                return False

            self.stoppingCriteria()

            if self.n_little_improvement >= 1 or self.stop < self.th_stop:
                print('Converged')
                return True

        return False


    def stoppingCriteria(self):
        if self.dV < 1e-12:
            self.n_little_improvement += 1
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
        self.direction = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.y = []
        self.s = []
        self.H0 = [np.eye(m.nu) for m in self.problem.runningModels]
        self.r = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.rho = np.zeros(self.memory_length)
        self.aux0 = np.zeros(self.memory_length)
        self.aux1 = 0
        #self.curvature = 0
        self.curvature_curr = 0.
        self.curvature_prev = 0.
        self.curvature_0 = 0.
        self.alpha = 0.
        self.alpha_p = 0.
        self.alpha_lo = 0.
        self.alpha_hi = 0.
        self.alpha_max = 1.
        self.curvature_lo = 0.
        self.curvature_hi = 0.
        self.cost_try_lo = 0.
        self.cost_try_hi = 0.

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




