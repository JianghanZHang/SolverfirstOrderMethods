import numpy as np
from numpy import linalg

import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract
import pdb
import time

DEBUGGING = False
VERBOSE = False

def rev_enumerate(l):
    return reversed(list(enumerate(l)))

def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error

class SolverLBGFS(SolverAbstract):
    def __init__(self, shootingProblem, memory_length = 30):
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
        self.th_stop = 1e-5
        self.n_little_improvement = 0
        self.c1 = 1e-4
        self.c2 = 0.8
        # self.c = 1e-4
        self.past_grad = 0.
        self.curr_grad = 0.
        self.change = 0.
        self.change_p = 0.
        self.lb = 0.
        self.ub = 0.
        self.memory_length = memory_length
        self.alpha_threshold = 1e-10
        self.numIter = 0
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

    def computeDirection(self, numIter, recalc=True):

        self.direction_p = self.direction.copy()
        self.dJdu_p = self.dJdu.copy()

        if recalc:
            if DEBUGGING: print("Going into Calc from compute direction")

        self.calc()
        if DEBUGGING: print("Going into Backward Pass from compute direction")
        self.backwardPass(numIter)  # get new dJdu

        self.q = self.dJdu.copy()
        q_flat = self.q.flatten()

        for i in range(min(self.memory_length, numIter) - 1, -1, -1):
            self.rho[i] = 1 / (self.y_flat[i].T @ self.s_flat[i])
            self.aux0[i] = self.rho[i] * (self.s_flat[i].T @ q_flat)
            q_flat -= self.aux0[i] * self.y_flat[i]

        H_init = self.init_hessian_approx(numIter)  # Previous y is y[-2], previous s is s[-1]

        r_flat = H_init * q_flat
        if VERBOSE: print(f'norm of r_init: {np.linalg.norm(r_flat)}.')

        for i in range(0, min(self.memory_length, numIter), 1):
            aux1 = self.rho[i] * (self.y_flat[i].T @ r_flat)
            r_flat += (self.aux0[i] - aux1) * self.s_flat[i]

        self.direction = -r_flat.reshape(self.q.shape)
        self.directions.append(np.linalg.norm(self.direction))
        if VERBOSE: print(f'norm of direction: {np.linalg.norm(self.direction)}.')

    def backwardPass(self, num_iter):

        self.dJdx[-1, :] = self.problem.terminalData.Lx
        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu[t, :] = data.Lu + self.dJdx[t + 1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t + 1, :] @ data.Fx

        if num_iter != 0:
            if num_iter - 1 < self.memory_length:
                self.y_flat[num_iter - 1] = (self.dJdu - self.dJdu_p).flatten()

            else:
                self.y_flat[:-1] = self.y_flat[1:]
                self.y_flat[-1] = (self.dJdu - self.dJdu_p).flatten()

        self.Qu = self.dJdu
        self.kkt = linalg.norm(self.dJdu, 2)
        self.KKTs.append(self.kkt)

    def init_hessian_approx(self, num_iter):
        K = 10
        r = .5
        if num_iter < 1:
            gamma = 1
            self.gammas.append(gamma)
            return gamma

        elif num_iter < self.memory_length:
            num = self.y_flat[num_iter-1].T @ self.s_flat[num_iter-1]  # should be a scalar
            den = self.y_flat[num_iter-1].T @ self.y_flat[num_iter-1]  # should be a scalar
            gamma = num / den
            self.gammas.append(gamma)
            return K * gamma ** r
        else:
            num = self.y_flat[-1].T @ self.s_flat[-1]  # should be a scalar
            den = self.y_flat[-1].T @ self.y_flat[-1]  # should be a scalar
            gamma = num / den
            self.gammas.append(gamma)
            return K * gamma ** r

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
            # model.calcDiff(data, self.xs_try[t], self.us_try[t])
            self.xs_try[t + 1] = data.xnext
            cost_try += data.cost

            # For Wolfe condition (sufficient decrease)
            self.threshold += (self.c1 * self.dJdu[t, :].T @ (alpha * self.direction[t, :]))
            self.curvature_0 += self.dJdu[t, :].T @ self.direction[t, :]

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])
        assert self.curvature_0 <= 0
        cost_try += self.problem.terminalData.cost

        return cost_try

    def tryStep(self, numIter):
        satisfied = self.lineSearch(numIter)
        return satisfied

    def lineSearch(self, numIter):

        if DEBUGGING: print(f'Going into forwardPass from lineSearch initialization using alpha = current_alpha.')
        self.cost_try = self.forwardPass(self.alpha)
        self.curvature_curr = self.calcCurvature()

        self.alpha_max = 1.
        self.alpha_p = 0.
        self.k = 10
        self.alpha = 2 ** (-self.k)

        cost_try_max = self.forwardPass(self.alpha_max)
        curvature_max = self.calcCurvature()

        if numIter >= 1: #self.memory_length:
            self.guess = (2 * (self.cost - self.cost_p) / self.curvature_0)
            #self.guess = self.alpha_prevIter * (self.curvature_prev / self.curvature_0)
            #self.guess = 1
            self.guesses.append(self.guess)
            if VERBOSE: (f'guess= {self.guess}')
            self.alpha = self.guess

        # Computing costs and curvatures for alpha_1 and alpha_0 before going into the loop
        self.cost_try_p = self.cost
        self.curvature_prev = self.curvature_0

       # self.cost_try = self.forwardPass(self.alpha)
       # self.curvature_curr = self.calcCurvature()

        for i in range(0, self.k + 2):

            self.cost_try = self.forwardPass(self.alpha)
            # Computing new cost(alpha_i), now us_try=us + alpha_i*direction, and (xs_try, us_try) is feasible.
            self.curvature_curr = self.calcCurvature()
            # Computing new curvature(alpha_i), the curvature was computed using us_try=us + alpha_i*direction.

            # if DEBUGGING:
            #     print(f'in iteration {i}:')
            #     print(f'current_alpha: {self.alpha}, direction:{self.direction}')
            #     print(f'cost_try: {self.cost_try}; curvature(current_alpha): {self.curvature_curr}')
            #     print(f'cost: {self.cost}; curvature(0): {self.curvature_0}')
            #     print(f'cost_try_previous: {self.cost_try_p}')

            if self.cost_try > self.cost + self.c1 * self.alpha * self.curvature_0 or \
                    (self.cost_try >= self.cost_try_p and i > 1) or np.isnan(self.cost_try):
                # sufficient decrease was not satisfied
                if i != 0:
                    #if DEBUGGING: print(f'going into zoom because sufficient decrease condition failed.')
                    return self.zoom(reversed=False)
                else:
                    #if DEBUGGING: print(f'reset alpha')
                    self.alpha_p = 0
                    self.alpha = 2 ** (-self.k)
                    continue

            if abs(self.curvature_curr) <= -self.c2 * self.curvature_0:  # curvature(alpha = 0) is from forwardPass
                #if VERBOSE: print('line search succeed.')
                self.alpha_prevIter = self.alpha
                # current alpha satisfy Wolfe condition -> stop line search
                return True

            if DEBUGGING: print(f'curvature condition failed.')

            if self.curvature_curr >= 0:
                # alphas_i overshoot -> going into reversed zoom
                if i != 0:
                    #if DEBUGGING: print(f'in iteration {i}, going into zoom because current curvature is positive.')
                    return self.zoom(reversed=True)
                else:
                    #if DEBUGGING: print(f'reset alpha')
                    self.alpha_p = 0
                    self.alpha = 2 ** (-self.k)
                    continue

            self.alpha_p = self.alpha
            self.alpha *= 2
            self.cost_try_p = self.cost_try  # record cost(alpha_i-1) before computing new cost
            self.curvature_prev = self.curvature_curr  # record curvature(alpha_i-1)

        return False

    def zoom(self, reversed):
        # initialize the algorithm
        if reversed:  # This is the case: zoom(alpha_i, alpha_i-1)
            alpha_lo = self.alpha
            alpha_hi = self.alpha_p
            curvature_lo = self.curvature_curr
            curvature_hi = self.curvature_prev
            cost_try_lo = self.cost_try
            cost_try_hi = self.cost_try_p
        else:  # This is the case: zoom(alpha_i-1, alpha_i)
            alpha_lo = self.alpha_p
            alpha_hi = self.alpha
            curvature_lo = self.curvature_prev
            curvature_hi = self.curvature_curr
            cost_try_lo = self.cost_try_p
            cost_try_hi = self.cost_try

        for i in range(0, self.k + 1):
            self.alpha = self.cubicInterpolation(alpha_lo, alpha_hi, curvature_lo, curvature_hi, cost_try_lo,
                                                 cost_try_hi)

            self.cost_try = self.forwardPass(self.alpha)  # now us_try = us + alpha * direction
            self.curvature_curr = self.calcCurvature()

            if DEBUGGING:
                print(
                    f'in zoom, current interpolated alpha: {self.alpha}. With [alpha_lo, alpha_hi] = [{alpha_lo}, {alpha_hi}]')
                print(f'in iteration {i}:')
                print(f'current_alpha: {self.alpha}')
                print(f'cost_try: {self.cost_try}; curvature(current_alpha): {self.curvature_curr}')
                print(f'cost: {self.cost}; curvature(0): {self.curvature_0}')

            if self.cost_try > self.cost + self.c1 * self.alpha * self.curvature_0 or self.cost_try >= cost_try_lo:
                if DEBUGGING:
                    print(f'sufficient decrease condition was not satisfied in zoom, changing upper bound of alpha to current alpha.')
                alpha_hi = self.alpha
                curvature_hi = self.curvature_curr
                cost_try_hi = self.cost_try

            else:
                if abs(self.curvature_curr) <= -self.c2 * self.curvature_0:
                    # current alpha satisfy the Wolfe condition -> stop line search
                    if VERBOSE: print(f'line search -> zoom succeed.')
                    self.alpha_prevIter = self.alpha
                    return True

                if self.curvature_curr * (alpha_hi - alpha_lo) >= 0:
                    if DEBUGGING: print(f'reversing interval of alpha')
                    alpha_hi = alpha_lo
                    curvature_hi = curvature_lo
                    cost_try_hi = cost_try_lo

                alpha_lo = self.alpha
                curvature_lo = self.curvature_curr
                cost_try_lo = self.cost_try
                if DEBUGGING: print(f'sufficient decrease condition was satisfied, '
                      f'but the curvature condition was not -> changing lower bound of alpha to current alpha.')
        return False

    def cubicInterpolation(self, alpha_l, alpha_r, curvature_l, curvature_r, cost_try_l, cost_try_r):
        # Note: it is possible to have alpha_i < alpha_i-1

        #return min(alpha_l, alpha_r) + .1 * max(alpha_r, alpha_l)

        if DEBUGGING: print('in cubicInterpolation:')

        d1 = curvature_l + curvature_r - 3 * ((cost_try_l - cost_try_r) / (alpha_l - alpha_r))
        d2 = np.sign(alpha_r - alpha_l) * ((d1**2 - curvature_l * curvature_r) ** .5)

        if DEBUGGING:
            print(f'd1: {d1}, d2: {d2}')
            print(f'alpha_l: {alpha_l}, alpha_r: {alpha_r}, curvature_l: {curvature_l}, curvature_r: {curvature_r}, cost_try_l: {cost_try_l}, cost_try_r: {cost_try_r}')
        alpha = alpha_r - (alpha_r - alpha_l) * ((curvature_r + d2 - d1) / (curvature_r - curvature_l + 2 * d2))

        if abs(alpha - alpha_l) < 1e-8 or abs(alpha - alpha_r) < 1e-8 or np.isnan(d1) or np.isnan(d2):
            if DEBUGGING: print(f'bad interpolation, using a safeguarded alpha')
            return max(alpha_l, alpha_r) / 2

        return alpha

    def solve(self, init_xs=None, init_us=None, maxIter=10000, isFeasible=True):
        # ___________________ Initialize ___________________#
        if init_xs is None:
            init_xs = [np.zeros(m.state.nx) for m in self.models()]
        if init_us is None:
            init_us = [np.zeros(m.nu) for m in self.problem.runningModels]

        init_xs[0][:] = self.problem.x0.copy()  # Initial condition guess must be x0
        self.xs_try[0][:] = self.problem.x0.copy()

        if not isFeasible:
            init_xs = self.problem.rollout(init_us)

        self.setCandidate(init_xs, init_us, True)

        self.cost = self.calc()  # self.forwardPass(1.)  # compute initial value for merit function
        self.costs.append(self.cost)

        if VERBOSE: print("initial cost is %s" % self.cost)

        self.guess_accepted = []

        for i in range(maxIter):
            start_time = time.time()
            self.numIter = i
            recalc = True  # this will recalculate derivatives in computeDirection
            while True:  # backward pass
                try:
                    self.computeDirection(i, recalc=True)

                except:
                    print('In', i, 'th iteration.')
                    raise BaseException("Backward Pass Failed")
                break

            # if self.n_little_improvement >= 1 or self.kkt < self.th_stop:
            if self.kkt < self.th_stop:
                print('Converged')
                return True

            while True:  # Going into line search
                try:
                    if VERBOSE: print(f'######################## Going into tryStep @ iteration {i} ##########################')
                    satisfied = self.tryStep(i)
                except:
                    # repeat starting from a smaller alpha
                    print("Try Step Failed for alpha = %s" % self.alpha)
                    raise BaseException("Forward Pass Failed")
                    continue
                break

            if satisfied:
                if VERBOSE:
                    print(f'in {i}th iteration:')
                    print("step accepted for alpha = %s \n new cost is %s" % (self.alpha, self.cost_try))
                self.alphas.append(self.alpha)
                if i < self.memory_length:  # keep track of the most recent m steps
                    self.s_flat[i] = (self.alpha * self.direction).flatten()
                else:
                    self.s_flat[:-1] = self.s_flat[1:]
                    self.s_flat[-1] = (self.alpha * self.direction).flatten()

                self.dV = self.cost - self.cost_try
                self.setCandidate(self.xs_try, self.us_try, isFeasible)
                self.cost_p = self.cost
                self.cost = self.cost_try
                self.costs.append(self.cost)

            else:
                if DEBUGGING: print('line search failed')
                return False

            if self.alpha == self.guess:
                self.guess_accepted.append(True)
            else:
                self.guess_accepted.append(False)


            self.stoppingCriteria()
            end_time = time.time()
            self.times.append(end_time - start_time)

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
        self.dJdu_try = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdu_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.q = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.dJdx_try = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.direction = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.directions = []
        self.y_flat = np.tile(np.array([np.zeros([m.nu]) for m in self.problem.runningModels]).flatten(),
                              (self.memory_length, 1))
        self.s_flat = np.tile(np.array([np.zeros([m.nu]) for m in self.problem.runningModels]).flatten(),
                              (self.memory_length, 1))
        self.H0 = [np.eye(m.nu) for m in self.problem.runningModels]
        self.r = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.rho = np.zeros(self.memory_length)
        self.rho1 = np.zeros(self.memory_length)
        self.aux0 = np.zeros(self.memory_length)
        self.aux01 = np.zeros(self.memory_length)
        self.aux1 = 0
        # self.curvature = 0
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
        self.alpha_prevIter = 2 ** (-10)
        self.cost_p = 0.
        self.h = 1.
        self.initial_alpha_accepted = []
        self.gamma_accepted = []
        self.guess_accepted = []
        self.costs = []
        self.alphas = []
        self.gammas = []
        self.guess = -1
        self.guesses = []
        self.KKTs = []
        self.kkt = 0.
        self.times = []

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
