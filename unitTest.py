import pdb

import crocoddyl
import numpy as np
from solverILQR import SolverILqr
from solverGD import SolverGD
#from solverLBFGS import SolverLBGFS
from solverLBGFS_vectorized import SolverLBGFS
import psutil
import time

from example import quadrotor_problem
from example import arm_manipulation_problem
from example import humanoid_taichi_problem
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, NX=6, NU=3, T=20, maxIter=100000):
        self.NX = NX
        self.NU = NU
        self.T = T
        self.maxIter = maxIter

        self.x0 = np.ones(NX) * 1.
        self.runningModel = crocoddyl.ActionModelLQR(NX, NU)
        self.terminalModel = crocoddyl.ActionModelLQR(NX, NU)
        self.problem_lqr = crocoddyl.ShootingProblem(self.x0, [self.runningModel] * T, self.terminalModel)

        self.problem_arm_manipulation = arm_manipulation_problem(self.T)
        self.problem_humanoid_taichi = humanoid_taichi_problem(self.T)
        self.problem_quadrotor = quadrotor_problem(self.T)

        self.init_us = None
        self.init_xs = None
        self.iLQR = None
        self.DDP = None
        self.GD = None
        self.LBFGS = None
        self.BGFS = None

    def testCrocoddylDDP(self, problem):
        self.DDP = crocoddyl.SolverDDP(problem)
        self.init_us = [np.zeros(m.nu) for m in problem.runningModels]
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.DDP.solve(self.init_xs, self.init_us, self.maxIter, True, 0)
        end_time = time.time()
        return end_time - start_time

    def testILQR(self, problem):
        self.iLQR = SolverILqr(problem)
        self.init_us = [np.zeros(m.nu) for m in problem.runningModels]
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.iLQR.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        return end_time - start_time

    def testGD(self, problem):
        self.GD = SolverGD(problem)
        self.init_us = [np.zeros(m.nu) for m in problem.runningModels]
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.GD.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        return end_time - start_time

    def testLBFGS(self, problem):
        self.LBFGS = SolverLBGFS(problem)
        self.init_us = [np.zeros(m.nu) for m in problem.runningModels]
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.LBFGS.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        return end_time - start_time

if __name__ == '__main__':
    T = 1
    tester = Tester(T=T)
    #problem = tester.problem_lqr
    problem = tester.problem_arm_manipulation
    # print('DDP testing:')
    # running_time1 = tester.testCrocoddylDDP(problem)

    print('iLQR testing:')
    running_time3 = tester.testILQR(problem)
    #pdb.set_trace()
    print('LBFGS testing:')
    running_time2 = tester.testLBFGS(problem)

    print(f'optimal control form iLQR solver: {tester.iLQR.us[0][:]}, cost= {tester.iLQR.cost}, '
          f'total number of iterations={tester.iLQR.numIter}')
    print(f'optimal control form LBFGS solver: {tester.LBFGS.us[0][:]}, cost= {tester.LBFGS.cost}, '
          f'total number of iterations={tester.LBFGS.numIter}, # initial guess of alpha accepted={len(tester.LBFGS.initial_alpha_accepted)}, '
          f'average magnitude of direction={sum(tester.LBFGS.directions)/len(tester.LBFGS.directions)}')

    start = 0
    end = -1
    # Set the figure size
    fig1, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 15))

    fig1.suptitle(f'L-BFGS Solver Metrics, T={T}', fontsize=16)

    color = 'tab:blue'
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(tester.LBFGS.costs[:], color=color, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('Iteration')  # Set the x-axis label
    ax1.grid(True)

    color = 'tab:red'
    ax2.set_ylabel('KKT(log10)', color=color)
    ax2.plot(np.log10(tester.LBFGS.KKTs[:]), color=color, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlabel('Iteration')  # Set the x-axis label
    ax2.grid(True)

    plt.savefig(f'plots/LBFGS(T={T}).png')

    fig2, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 15))

    fig2.suptitle(f'iLQR Solver Metrics, T={T}', fontsize=16)

    color = 'tab:blue'
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(tester.iLQR.costs, color=color, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('Iteration')  # Set the x-axis label
    ax1.grid(True)

    color = 'tab:red'
    ax2.set_ylabel('KKT(log10)', color=color)
    ax2.plot(np.log10(tester.iLQR.KKTs), color=color, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlabel('Iteration')  # Set the x-axis label
    ax2.grid(True)

    plt.savefig(f'plots/iLQR(T={T}).png')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    # print(f'optimal control form LBFGS solver: {tester.iLQR.us[0][:]}, cost= {tester.iLQR.cost}')
    # import pdb; pdb.set_trace()
    # alpha_accepted = [False] * (tester.LBFGS.numIter+1)
    # for i in tester.LBFGS.initial_alpha_accepted:
    #     alpha_accepted[i] = True
