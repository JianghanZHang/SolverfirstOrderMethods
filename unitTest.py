import pdb

import crocoddyl
import numpy as np
from solverILQR import SolverILqr
from solverGD import SolverGD
from solverADAM import SolverADAM
#from solverLBFGS import SolverLBGFS
from solverLBGFS_vectorized import SolverLBGFS
from solverBGFS import SolverBGFS
import psutil
import time
import threading
from quadrotor import quadrotor_problem
from arm_manipulation import arm_manipulation_problem
import matplotlib.pyplot as plt

class Tester:
    def __init__(self, NX=6, NU=3, T=10, maxIter=10000):
        self.NX = NX
        self.NU = NU
        self.T = T
        self.maxIter = maxIter

        # self.x0 = np.ones(NX) * 1.
        # self.runningModel = crocoddyl.ActionModelLQR(NX, NU)
        # self.terminalModel = crocoddyl.ActionModelLQR(NX, NU)
        # self.problem = crocoddyl.ShootingProblem(self.x0, [self.runningModel] * T, self.terminalModel)

        self.problem = arm_manipulation_problem(self.T)

        self.init_us = [np.zeros(m.nu) for m in self.problem.runningModels]
        self.init_xs = self.problem.rollout(self.init_us)
        self.iLQR = SolverILqr(self.problem)
        self.DDP = crocoddyl.SolverDDP(self.problem)
        self.GD = SolverGD(self.problem)
        self.ADAM = SolverADAM(self.problem)
        self.LBFGS = SolverLBGFS(self.problem)
        self.BGFS = SolverBGFS(self.problem)


    def testCrocoddylDDP(self):
        start_time = time.time()
        self.DDP.solve(self.init_xs, self.init_us, self.maxIter, True, 0)
        end_time = time.time()
        # time.sleep(5)
        return end_time - start_time

    def testILQR(self):
        start_time = time.time()
        self.iLQR.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        #time.sleep(5)
        return end_time - start_time

    def testGD(self):
        start_time = time.time()
        self.GD.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        # time.sleep(5)
        return end_time - start_time

    def testADAM(self):
        start_time = time.time()
        self.ADAM.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        # time.sleep(5)
        return end_time - start_time

    def testLBFGS(self):
        start_time = time.time()
        self.LBFGS.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        # time.sleep(5)
        return end_time - start_time

    def testBFGS(self):
        start_time = time.time()
        self.BGFS.solve(self.init_xs, self.init_us, self.maxIter)
        end_time = time.time()
        # time.sleep(5)
        return end_time - start_time



def monitor_threads(stop_event):
    current_pid = psutil.Process().pid
    while not stop_event.is_set():
        try:
            process = psutil.Process(current_pid)
            num_threads = process.num_threads()
            print(f'Number of threads: {num_threads}')
        except psutil.NoSuchProcess:
            print("Process has ended")
            break
        time.sleep(1)

def testGrad(Solver):
    Solver.numDiff_grad()
    print('xs:', Solver.xs)
    print('us:', Solver.us)
    #Solver.calc() # reclac
    #print('grad:', Solver.dJdu)
    Solver.backwardPass(1)
    print('xs:', Solver.xs)
    print('us:', Solver.us)
    print('analyticDIff_grad:', Solver.dJdu)

if __name__ == '__main__':
    tester = Tester()
    print('DDP testing:')
    running_time1 = tester.testCrocoddylDDP()

    print('LBFGS testing:')
    running_time2 = tester.testLBFGS()

    # print('iLQR testing:')
    # running_time3 = tester.testILQR()

    print(f'optimal control form DDP solver: {tester.DDP.us[0][:]}, cost= {tester.DDP.cost}')
    print(f'optimal control form LBFGS solver: {tester.LBFGS.us[0][:]}, cost= {tester.LBFGS.cost}, '
          f'total number of iterations={tester.LBFGS.numIter}')

    #import pdb; pdb.set_trace()
    alpha_accepted = [False] * (tester.LBFGS.numIter+1)
    for i in tester.LBFGS.initial_alpha_accepted:
        alpha_accepted[i] = True

    gamma_accepted = [False] * (tester.LBFGS.numIter + 1)
    for i in tester.LBFGS.gamma_accepted:
        gamma_accepted[i] = True

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True)

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(tester.LBFGS.costs, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    color = 'tab:red'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Alpha=initial guess accepted', color=color)
    ax2.plot(alpha_accepted, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['False', 'True'])

    # color = 'tab:green'
    # # we already handled the x-label with ax1
    # ax3.set_ylabel('Gamma accepted', color=color)
    # ax3.plot(gamma_accepted, color=color)
    # ax3.tick_params(axis='y', labelcolor=color)
    # ax3.set_yticks([0, 1])
    # ax3.set_yticklabels(['False', 'True'])

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax3.set_ylabel('gamma', color=color)
    ax3.plot(tester.LBFGS.gammas, color=color)
    ax3.tick_params(axis='y', labelcolor=color)

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax4.set_ylabel('alphas', color=color)
    ax4.plot(tester.LBFGS.alphas, color=color)
    ax4.tick_params(axis='y', labelcolor=color)



    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()
    pdb.set_trace()

    # print(f'optimal control form LBFGS solver: {tester.iLQR.us[0][:]}, cost= {tester.iLQR.cost}')