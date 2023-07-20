import pdb

import crocoddyl
import numpy as np
from solverILQR import SolverILqr
from solverGD import SolverGD
from solverNAG import SolverNAG
from solverLBGFS_vectorized import SolverLBGFS
from solverADAM import SolverADAM
import time

from example import quadrotor_problem
from example import arm_manipulation_problem
from example import humanoid_taichi_problem
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, NX=6, NU=3, T=30, maxIter=1000000):
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
        self.NAG = None
        self.ADAM = None

    def testCrocoddylDDP(self, problem):
        self.DDP = crocoddyl.SolverDDP(problem)
        self.init_us = problem.quasiStatic([problem.x0] * problem.T)
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.DDP.solve(self.init_xs, self.init_us, self.maxIter, True, 0)
        end_time = time.time()
        return end_time - start_time

    def testILQR(self, problem):
        self.iLQR = SolverILqr(problem)
        self.init_us = problem.quasiStatic([problem.x0] * problem.T)
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.iLQR.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        return end_time - start_time

    def testGD(self, problem):
        self.GD = SolverGD(problem)
        self.init_us = problem.quasiStatic([problem.x0] * problem.T)
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.GD.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        return end_time - start_time

    def testLBFGS(self, problem):
        self.LBFGS = SolverLBGFS(problem)
        self.init_us = problem.quasiStatic([problem.x0] * problem.T)
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.LBFGS.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        return end_time - start_time

    def testNAG(self, problem):
        self.NAG = SolverNAG(problem)
        self.init_us = problem.quasiStatic([problem.x0] * problem.T)
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.NAG.solve(self.init_xs, self.init_us, self.maxIter, alpha=.01)
        end_time = time.time()
        return end_time - start_time

    def testADAM(self, problem):
        self.ADAM = SolverADAM(problem)
        self.init_us = problem.quasiStatic([problem.x0] * problem.T)
        self.init_xs = problem.rollout(self.init_us)
        start_time = time.time()
        self.ADAM.solve(self.init_xs, self.init_us, self.maxIter, alpha=1)
        end_time = time.time()
        return end_time - start_time

if __name__ == '__main__':
    T = 20
    tester = Tester(T=T)

    problem = tester.problem_lqr

    tester.testILQR(problem)

    tester.testADAM(problem)

    print(f'optimal control form iLQR solver: {tester.iLQR.us[0][:]}, cost= {tester.iLQR.cost}, '
          f'total number of iterations={tester.iLQR.numIter}, # initial guess of alpha accepted={sum(tester.iLQR.guess_accepted)}')
    # print(f'optimal control form LBFGS solver: {tester.LBFGS.us[0][:]}, cost= {tester.LBFGS.cost}, '
    #       f'total number of iterations={tester.LBFGS.numIter}, # initial guess of alpha accepted={sum(tester.LBFGS.guess_accepted)}, '
    #       f'average magnitude of direction={sum(tester.LBFGS.directions)/len(tester.LBFGS.directions)}')
    print(f'optimal control form NAG solver: {tester.ADAM.us[0][:]}, cost= {tester.ADAM.cost}, '
          f'total number of iterations={tester.ADAM.numIter}')

    start = 0
    end = -1
    # Set the figure size
    fig1, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 15))

    fig1.suptitle(f'ADAM Solver Metrics, T={T}', fontsize=16)

    color = 'tab:blue'
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(tester.ADAM.costs[:], color=color, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('Iteration')  # Set the x-axis label
    ax1.grid(True)

    color = 'tab:red'
    ax2.set_ylabel('KKT(log10)', color=color)
    ax2.plot(np.log10(tester.ADAM.KKTs[:]), color=color, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlabel('Iteration')  # Set the x-axis label
    ax2.grid(True)

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

