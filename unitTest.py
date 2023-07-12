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
    def __init__(self, NX=6, NU=3, T=20, maxIter=10000):
        self.NX = NX
        self.NU = NU
        self.T = T
        self.maxIter = maxIter

        self.x0 = np.ones(NX) * 1.
        self.runningModel = crocoddyl.ActionModelLQR(NX, NU)
        self.terminalModel = crocoddyl.ActionModelLQR(NX, NU)
        self.problem = crocoddyl.ShootingProblem(self.x0, [self.runningModel] * T, self.terminalModel)

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

    print('iLQR testing:')
    running_time3 = tester.testILQR()

    print(f'optimal control form iLQR solver: {tester.iLQR.us[0][:]}, cost= {tester.iLQR.cost}, '
          f'total number of iterations={tester.iLQR.numIter}')
    print(f'optimal control form LBFGS solver: {tester.LBFGS.us[0][:]}, cost= {tester.LBFGS.cost}, '
          f'total number of iterations={tester.LBFGS.numIter}, # initial guess of alpha accepted={len(tester.LBFGS.initial_alpha_accepted)}, '
          f'average magnitude of direction={sum(tester.LBFGS.directions)/len(tester.LBFGS.directions)}')

    #import pdb; pdb.set_trace()
    alpha_accepted = [False] * (tester.LBFGS.numIter+1)
    for i in tester.LBFGS.initial_alpha_accepted:
        alpha_accepted[i] = True

    import matplotlib.pyplot as plt
    start = 0
    end = -1
    # Set the figure size
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(10, 15))

    fig.suptitle('L-BFGS Solver Metrics', fontsize=16)

    color = 'tab:blue'
    ax1.set_ylabel('Cost', color=color)
    ax1.plot(tester.LBFGS.costs[start:end], color=color, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('Iteration')  # Set the x-axis label
    ax1.grid(True)

    color = 'tab:red'
    ax2.set_ylabel('Alpha=initial guess accepted', color=color)
    ax2.plot(alpha_accepted[start:end], color=color, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['False', 'True'])
    ax2.set_xlabel('Iteration')  # Set the x-axis label
    ax2.grid(True)

    color = 'tab:green'
    ax3.set_ylabel('Direction Magnitude', color=color)
    ax3.plot(tester.LBFGS.directions[start:end], color=color, linestyle='-')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_xlabel('Iteration')  # Set the x-axis label
    ax3.grid(True)

    color = 'tab:orange'
    ax4.set_ylabel('Alphas', color=color)
    ax4.plot(tester.LBFGS.alphas[start:end], color=color, linestyle='-')
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.set_xlabel('Iteration')  # Set the x-axis label
    ax4.grid(True)

    color = 'tab:purple'
    ax5.set_ylabel('Gamma', color=color)
    ax5.plot(tester.LBFGS.gammas[start:end], color=color, linestyle='-')
    ax5.tick_params(axis='y', labelcolor=color)
    ax5.set_xlabel('Iteration')  # Set the x-axis label
    ax5.grid(True)

    # Display the plot
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    pdb.set_trace()

    # print(f'optimal control form LBFGS solver: {tester.iLQR.us[0][:]}, cost= {tester.iLQR.cost}')