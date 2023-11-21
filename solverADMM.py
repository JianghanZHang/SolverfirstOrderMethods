import numpy as np
from numpy import linalg
import scipy.linalg as scl
import crocoddyl
from crocoddyl import SolverAbstract
import json
import pdb

# Load the configuration from the JSON file
with open("ADMMconfig.json", 'r') as file:
    CONFIG = json.load(file)

LINE_WIDTH = 100

VERBOSE = False



def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error


class SolverADMM(SolverAbstract):
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

    # def dynamicInfeasibility(self):
    #     # compute dynamic infeasibility
    #     #pdb.set_trace()
    #     self.dynInf = np.array([np.linalg.norm(self.models()[t].calc(self.xs_try[t], self.us_try[t]).xnext) for t in range(len(self.us_try))])
    #     self.dynInf = np.linalg.norm(self.dynInf, 2)
    #     return self.dynInf

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod

    def calc(self):
        # compute cost and derivatives at deterministic nonlinear trajectory

        self.problem.calc(self.xs, self.us)
        cost = self.problem.calcDiff(self.xs, self.us)
        return cost
    
    def getCost(self):
        # pdb.set_trace()
        xs_temp = self.problem.rollout(self.us_try)
        # self.setCandidate(xs_temp, self.us)
        cost = self.problem.calc(xs_temp, self.us_try)
        return cost

    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        # self.dJdu_p = self.dJdu
        self.backwardPass()

    def backwardPass(self):
        self.dJdx[-1, :] = self.problem.terminalData.Lx
        self.dLdx[-1, :] = self.problem.terminalData.Lx + (-self.lambdas[-1, :] + self.rho * self.constraints[-1]).T @ (-np.eye(self.problem.terminalData.Fx.shape[0]))

        for t, (model, data) in rev_enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            self.dJdu[t, :] = data.Lu + self.dJdx[t+1, :] @ data.Fu
            self.dJdx[t, :] = data.Lx + self.dJdx[t+1, :] @ data.Fx

            # pdb.set_trace()
            # import pdb; pdb.set_trace()
            # self.dLdu[t, :] = self.dJdu[t, :] + (-self.lambdas[t, :] + self.miu * self.constraints[t]).T @ data.Fu
            # self.dLdx[t, :] = self.dJdx[t, :] + (-self.lambdas[t, :] + self.miu * self.constraints[t]).T @ data.Fx
            self.dLdu[t, :] = data.Lu + (-self.lambdas[t+1, :] + self.rho * self.constraints[t+1, :]).T @ data.Fu
            self.dLdx[t, :] = data.Lx + (-self.lambdas[t+1, :] + self.rho * self.constraints[t+1, :]).T @ data.Fx + (-self.lambdas[t-1, :] + self.rho * self.constraints[t, :]).T @ -np.eye(data.Fx.shape[0]) 

    
            # if t == 0:
            #     self.dLdx[t, :] = data.Lx + (-self.lambdas[t, :] + self.miu * self.constraints[t]).T @ (data.Fx)
            
            # else:
            #     self.dLdx[t, :] = data.Lx + (-self.lambdas[t, :] + self.miu * self.constraints[t]).T @ (data.Fx - np.eye(data.Fx.shape[0]))
               
        # For the Augmented Lagrangian    
        # self.Q = np.hstack((self.dLdx[1:], self.dLdu))
        self.Q = self.dLdu.copy()
        self.gradientNorm = linalg.norm(self.Q, 2)
        self.gradientNorms.append(self.gradientNorm)

        # For the cost function
        self.kkt = linalg.norm(np.hstack((self.Q, self.constraints[1:])), 2)
        self.KKTs.append(self.kkt)


    def forwardPass(self, alpha, i):
        cost_try = 0.

        # pdb.set_trace()
        ############################### ADAM #################################
        self.m_u = self.Beta1 * self.m_u + (1 - self.Beta1) * (self.dLdu)
        self.v_u = self.Beta2 * self.v_u + (1 - self.Beta2) * (self.dLdu**2)
        self.m_x = self.Beta1 * self.m_x + (1 - self.Beta1) * (self.dLdx)
        self.v_x = self.Beta2 * self.v_x + (1 - self.Beta2) * (self.dLdx**2)
        # if self.bias_correction:
        #     m_corrected = self.m / (1 - self.Beta1 ** (i + 2))
        #     v_corrected = self.v / (1 - self.Beta2 ** (i + 2))
        # else:
        #     m_corrected = self.m
        #     v_corrected = self.v
        self.update_u = -self.m_u / (np.sqrt(self.v_u) + self.eps)
        # self.update_x = -self.m_x / (np.sqrt(self.v_x) + self.eps)
        us = np.array(self.us)
        xs = np.array(self.xs)
        us_try = us + alpha * self.update_u
        xs_try = xs.copy()
        # xs_try[1:] -= alpha * update_x[1:]

        self.us_try = list(us_try)
        self.xs_try = list(xs_try)
        ####################################################################
        # pdb.set_trace()
        ############################## NAG #################################
        # self.lookahead = [np.array(a + self.miu * (a - b)) for a, b in zip(self.us_try, self.us_try_p)]

        # self.us_try = self.lookahead - self.alpha * self.dLdu
        # self.us_try = list(self.us_try)

        # self.us_try_p = self.us_try.copy()
        ####################################################################

        
        self.accumulated_update_u += alpha * self.update_u

        # if i % 10 == 0 and i != 0 :
        #     pdb.set_trace()

        self.u_magnitude.append(np.linalg.norm(self.us, 2))
        # need to make sure self.xs_try[0] = x0
        for t, (model, data) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            # model.calc(data, self.xs_try[t], self.us_try[t])
           
        
            if i % 10 == 0 and i != 0:
                self.update_x[t+1] = (data.Fx @ self.update_x[t] + data.Fu @ (alpha * self.accumulated_update_u[t]) + self.gap[t+1])
                self.xs_try[t+1] = self.xs[t+1] + self.update_x[t+1]
                model.calc(data, self.xs_try[t], self.us_try[t])
                self.gap[t+1] = data.xnext - self.xs_try[t+1]

                self.accumulated_update_u[t] = 0.0
            
            cost_try += data.cost
            self.constraints[t+1] = data.xnext - self.xs_try[t+1]  # c[t](x[t], u[t], x[t+1]) = f(x[t], u[t]) - x[t+1]
            
            # self.constraints[t] = self.xs_try[t+1] - data.xnext  # c[t+1] = x[t+1] - f(x[t], u[t]) 
            cost_try += (-self.lambdas[t, :] + .5 * self.rho * self.constraints[t]).T @ self.constraints[t] 

        
        if i % 10 == 0 and i != 0:
            self.lambdas[1:] -= self.rho * self.constraints[1:]

        self.problem.terminalModel.calc(self.problem.terminalData, self.xs_try[-1])
        
        cost_try += self.problem.terminalData.cost

        self.Infeasibilities.append(np.linalg.norm(self.constraints, 2))

        return cost_try

    def tryStep(self, alpha, i):
        # self.direction_p = self.direction
        self.cost_try = self.forwardPass(alpha, i)

        return self.cost - self.cost_try
    
    def updatingALM(self):
        print(f'Terminated, updating ALM parameters at {self.inner_iter}th iteration')
        # pdb.set_trace()
        self.lambdas[1:] -= self.rho * self.constraints[1:]
        # print(f"lambda[0] before update: {self.lambdas[0]}")

        # self.lambdas = np.array([lam - self.miu * constraint for lam, constraint in zip(self.lambdas, self.constraints)])

        # print(f"constraints[0]: {self.constraints[0]}")
        # print(f"lambda[0] after update: {self.lambdas[0]}")
        # import pdb; pdb.set_trace()
        self.rho = self.gamma * self.rho

        self.gradientThreshold *= CONFIG["gradientThresholdDecay"]
        self.constraintThreshold *= CONFIG["constraintThresholdDecay"]

    def checkUpdatingandConvergence(self):
        # import pdb; pdb.set_trace()
        if self.kkt < 1e-4 and linalg.norm(self.constraints, 2) < 1e-3:
            print("Converged!")
            return True

        # if self.gradientNorm < self.gradientThreshold:
        if self.inner_iter % 100 == 0:
            self.outer_iter += 1
            self.updatingALM()
            print("Updating ALM parameters")

        return False
        # if linalg.norm(self.constraints, 2) < self.constraintThreadshold:
        #     self.outer_iter += 1
        #     self.updatingALM()


    def solve(self, init_xs=None, init_us=None, maxOuterIter=CONFIG["maxOuterIterations"], maxInnerIter=CONFIG["maxInnerIterations"], isFeasible=False):
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

        self.cost = self.calc()  # self.forwardPass(1.)  # compute initial value for merit function
        self.costs.append(self.cost)

        #print("initial cost is %s" % self.cost)
        print(f'maxInnerIter={maxInnerIter}, maxOuterIter={maxOuterIter}')

        while self.outer_iter < maxOuterIter and self.inner_iter < maxInnerIter:

            # if self.inner_iter % 20000 == 0:
            #     xs_temp = self.problem.rollout(self.us)
            #     self.setCandidate(xs_temp, self.us)
            #     cost = self.problem.calc(xs_temp, self.us)
            self.inner_iter += 1
            recalc = True  # this will recalculate derivatives in computeDirection
            while True:  # backward pass
                try:
                    self.computeDirection(recalc=recalc)

                except:
                    print('In', self.inner_iter, 'th iteration.')

                    #pdb.set_trace()
                    raise BaseException("Backward Pass Failed")
                break

            
            while True:  # forward pass with line search
                try:
                    self.tryStep(self.alpha, self.inner_iter)

                except:
                    # repeat starting from a smaller alpha
                    print("Try Step Failed for alpha = %s" % self.alpha)
                    raise BaseException("Forward Pass Failed")

                break
        
            self.setCandidate(self.xs_try, self.us_try)

            # converged = self.checkUpdatingandConvergence()

            self.cost = self.getCost()
            self.costs.append(self.getCost())

            # if converged: return True

        return False

    def stoppingCriteria(self):
        if self.dV < 1e-12:
            self.n_little_improvement += 1
            if VERBOSE: print('Little improvement.')

    def allocateData(self):
        
        self.xs_try = [np.zeros(m.state.nx) for m in self.models()]
        self.xs_try[0][:] = self.problem.x0.copy()
        self.us_try = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.lookahead = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.us_try_p = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.dLdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dLdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.dJdu = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.dJdx = np.array([np.zeros(m.state.ndx) for m in self.models()])
        self.lambdas = np.array([np.zeros(m.state.nx) for m in self.models()])
        self.gap = np.array([np.zeros(m.state.ndx) for m in self.models()])
        
        self.update_u = np.array([np.zeros(m.nu) for m in self.problem.runningModels])
        self.update_x = np.array([np.zeros(m.state.ndx) for m in self.models()])
        
        self.m_x = np.zeros_like(self.dLdx)
        self.v_x = np.zeros_like(self.dLdx)
        self.m_u = np.zeros_like(self.dJdu)
        self.v_u = np.zeros_like(self.dJdu)
        self.rho = CONFIG["rho"]
        self.gamma = CONFIG["gamma"]
        self.miu = 0.1
        self.Beta1 = .9
        self.Beta2 = .999
        self.eps = 1e-8
        self.outer_iter = 1
        # self.objective_cost = 0.
        self.constraintThreshold = CONFIG["constraintThreshold"]
        self.gradientThreshold = CONFIG["gradientThreshold"]
        self.constraints = np.array([np.zeros(m.state.nx) for m in self.models()])
        self.dynInf = 0.
        self.alpha_p = 0.
        self.alpha = CONFIG["alpha"]
        self.dJdu_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.direction_p = np.array([np.zeros([m.nu]) for m in self.problem.runningModels])
        self.inner_iter = 1
        self.costs = []
        self.Infeasibilities = []
        self.gradientNorms = []
        self.kkt = 0.
        self.KKTs = []
        self.u_magnitude = []
        self.accumulated_update_u = np.array([np.zeros(m.nu) for m in self.problem.runningModels])
        # pdb.set_trace()

