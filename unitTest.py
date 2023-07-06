import crocoddyl
import numpy as np
from solverILQR import SolverILqr
from solverGD import SolverGD
from solverADAM import SolverADAM
from solverLBFGS import SolverLBGFS
from solverBGFS import SolverBGFS
import psutil
import time
import threading
from quadrotor import quadrotor_problem
from arm_manipulation import arm_manipulation_problem

class Tester:
    def __init__(self, NX=6, NU=1, T=10, maxIter=100000):
        self.NX = NX
        self.NU = NU
        self.T = T
        self.maxIter = maxIter

        self.x0 = np.ones(NX) * 10.
        self.runningModel = crocoddyl.ActionModelLQR(NX, NU)
        self.terminalModel = crocoddyl.ActionModelLQR(NX, NU)
        self.problem = crocoddyl.ShootingProblem(self.x0, [self.runningModel] * T, self.terminalModel)

        #self.problem = arm_manipulation_problem(self.T)

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
    stop_event_1 = threading.Event()
    monitor_thread_1 = threading.Thread(target=monitor_threads, args=(stop_event_1,))
    monitor_thread_1.start()
    running_time1 = tester.testCrocoddylDDP()
    stop_event_1.set()
    monitor_thread_1.join()


    print('LBFGS testing:')
    stop_event_2 = threading.Event()
    monitor_thread_2 = threading.Thread(target=monitor_threads, args=(stop_event_2,))
    monitor_thread_2.start()
    running_time2 = tester.testLBFGS()
    stop_event_2.set()
    monitor_thread_2.join()


    print('optimal control form DDP solver:', tester.DDP.us[0][:], 'cost=', tester.DDP.cost)
    print('optimal control from LBGFS solver:', tester.LBFGS.us[0][:], 'cost=', tester.LBFGS.cost)
    #print('optimal control form BGFS solver:', tester.BGFS.us[0][:], 'cost=', tester.BGFS.cost)
