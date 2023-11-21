import pdb
import crocoddyl
import numpy as np
from solverILQR import SolverILqr
from solverGD_backTracking import SolverGD
from solverNAG_lineSearch import SolverNAG
from solverLBGFS_vectorized import SolverLBGFS
from solverADAM_lineSearch import SolverADAM
from solverADAN import SolverADAN
from solverALM import SolverALM
from SolverMultipleShooting_lineSearch import SolverMSls
from solverMultipleShooting import SolverMS
from solverADMM import SolverADMM
import time

from example import quadrotor_problem
from example import arm_manipulation_problem
from example import humanoid_taichi_problem
import matplotlib.pyplot as plt
import pdb
import json
import os

with open("ALMconfig.json", 'r') as file:
    CONFIG = json.load(file)

class Tester:
    def __init__(self, NX=1, NU=1, T=20, maxIter=10):
        self.NX = NX
        self.NU = NU
        self.T = T

        # self.x0 = np.ones(NX) * 1.
        self.x0 = np.random.uniform(low=-2 * np.pi, high= 2 * np.pi, size=(NX, 1))
        self.runningModel = crocoddyl.ActionModelLQR(NX, NU)
        self.terminalModel = crocoddyl.ActionModelLQR(NX, NU)
        self.problem_lqr = crocoddyl.ShootingProblem(self.x0, [self.runningModel] * T, self.terminalModel)

        self.problem_arm_manipulation = arm_manipulation_problem(self.T)
        self.problem_humanoid_taichi = humanoid_taichi_problem(self.T)
        self.problem_quadrotor = quadrotor_problem(self.T)

        self.problem = None

    def test(self, solver, params):

        
        init_us = self.problem.quasiStatic([self.problem.x0] * self.problem.T)
        init_xs = self.problem.rollout(init_us)
        self.init_us = init_us

        if solver == 'DDP':

            DDP = crocoddyl.SolverDDP(self.problem)
    
            DDP.setCallbacks(
            [
                crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose(),
                # crocoddyl.CallbackDisplay(display),
            ]
            )
            # print("entering DDP solver")
            start_time = time.time()
            DDP.solve(init_xs, init_us, params["maxIter"], False, 1e-3)
            end_time = time.time()

            log = DDP.getCallbacks()[0]
            # crocoddyl.plotOCSolution(log.xs, log.us)
            # crocoddyl.plotConvergence(
            #     log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps
            # )
            print(f'optimal control form DDP solver: cost= {DDP.cost},'
        f'\nus[0]:{DDP.us[0]}, xs[0]:{DDP.xs[0]}\nus[-1]:{DDP.us[-1]}, xs[-1]:{DDP.xs[-1]}\n\n')
            
        elif solver == 'FDDP':

            self.FDDP = crocoddyl.SolverFDDP(self.problem)
    
            self.FDDP.setCallbacks(
            [
                crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose(),
                # crocoddyl.CallbackDisplay(display),
            ]
            )
            # print("entering DDP solver")
            start_time = time.time()
            self.FDDP.solve(init_xs, init_us, params["maxIter"], False, 1e-3)
            end_time = time.time()

            log = self.FDDP.getCallbacks()[0]
            # crocoddyl.plotOCSolution(log.xs, log.us)
            # crocoddyl.plotConvergence(
            #     log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps
            # )
            print(f'optimal control form FDDP solver: cost= {self.FDDP.cost},'
        f'\nus[0]:{self.FDDP.us[0]}, xs[0]:{self.FDDP.xs[0]}\nus[-1]:{self.FDDP.us[-1]}, xs[-1]:{self.FDDP.xs[-1]}\n\n')

        elif solver == "iLQR":
            iLQR = SolverILqr(self.problem)
            start_time = time.time()
            iLQR.solve(init_xs, init_us, params["maxIter"])
            end_time = time.time()
            print(f'optimal control form ILQR solver: cost= {iLQR.cost},'
        f'\nus[0]:{iLQR.us[0]}, xs[0]:{iLQR.xs[0]}\nus[-1]:{iLQR.us[-1]}, xs[-1]:{iLQR.xs[-1]}\n\n')
        
        elif solver == "GD":

            GD = SolverGD(self.problem)
            start_time = time.time()
            GD.solve(init_xs, init_us, params["maxIter"])
            end_time = time.time()
            print(f'optimal control form Gradient Descent solver: cost= {GD.cost},'
        f'\nus[0]:{GD.us[0]}, xs[0]:{GD.xs[0]}\nus[-1]:{GD.us[-1]}, xs[-1]:{GD.xs[-1]}\n\n')
        
        elif solver == "LBFGS":
            LBFGS = SolverLBGFS(self.problem)
            start_time = time.time()
            LBFGS.solve(init_xs, init_us, params["maxIter"])
            end_time = time.time()
            print(f'optimal control form LBFGS solver: cost= {LBFGS.cost},'
        f'\nus[0]:{LBFGS.us[0]}, xs[0]:{LBFGS.xs[0]}\nus[-1]:{LBFGS.us[-1]}, xs[-1]:{LBFGS.xs[-1]}\n\n')
        
        elif solver == "NAG":
            NAG = SolverNAG(self.problem)

            start_time = time.time()
            NAG.solve(init_xs, init_us, params["maxIter"])
            end_time = time.time()
            print(f'optimal control form NAG solver: cost= {NAG.cost},'
        f'\nus[0]:{NAG.us[0]}, xs[0]:{NAG.xs[0]}\nus[-1]:{NAG.us[-1]}, xs[-1]:{NAG.xs[-1]}\n\n')
        
        elif solver == "ADAM":

            ADAM = SolverADAM(self.problem)

            start_time = time.time()
            ADAM.solve(init_xs, init_us, params["maxIter"])
            end_time = time.time()
            print(f'optimal control form ADAM solver: cost= {ADAM.cost},'
        f'\nus[0]:{ADAM.us[0]}, xs[0]:{ADAM.xs[0]}\nus[-1]:{ADAM.us[-1]}, xs[-1]:{ADAM.xs[-1]}\n\n')
            
            fig1, (ax1, ax3, ax2) = plt.subplots(3, sharex=True, figsize=(10, 15))


            fig1.suptitle(f'SS Solver Metrics, max iterations={params["maxIter"]}, T={T}', fontsize=16)


            color = 'tab:blue'
            ax1.set_ylabel('Cost', color=color)
            ax1.plot(ADAM.costs[:], color=color, linestyle='-')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlabel('Iteration')  # Set the x-axis label
            ax1.grid(True)

            color = 'tab:red'
            ax3.set_ylabel('KKT(log10)', color=color)
            ax3.plot(np.log10(ADAM.KKTs[:]), color=color, linestyle='-')
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.set_xlabel('Iteration')  # Set the x-axis label
            ax3.grid(True)

            color = 'tab:red'
            ax2.set_ylabel('alpha', color=color)
            ax2.plot(ADAM.alphas[:], color=color, linestyle='-')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_xlabel('Iteration')  # Set the x-axis label
            ax2.grid(True)

             # Define the directory and filename

            folder_name = 'experiment/kuka_reaching_ocp'

            file_name = 'single_shooting.png'



            # Check if the folder exists, if not, create it

            if not os.path.exists(folder_name):

                os.makedirs(folder_name)



            # Full path

            full_path = os.path.join(folder_name, file_name)



            # Save the plot to the desired folder

            plt.savefig(full_path)

            
        
        elif solver == "ADAN":
            ADAN = SolverADAN(self.problem)
            ADAN.bias_correction = True

            start_time = time.time()
            ADAN.solve(init_xs, init_us, params["maxIter"], params["alpha"])
            end_time = time.time()
            print(f'optimal control form ADAN solver: cost= {ADAN.cost},'
        f'\nus[0]:{ADAN.us[0]}, xs[0]:{ADAN.xs[0]}\nus[-1]:{ADAN.us[-1]}, xs[-1]:{ADAN.xs[-1]}\n\n')
        
        elif solver == "ALM":
            self.ALM = SolverALM(self.problem)

            start_time = time.time()
            self.ALM.solve(init_xs, init_us)
            end_time = time.time()
            print(f'optimal control form ALM solver: cost= {self.ALM.cost},'
            f'\nus[0]:{self.ALM.us[0]}, xs[0]:{self.ALM.xs[0]}\nus[-1]:{self.ALM.us[-1]}, xs[-1]:{self.ALM.xs[-1]}\n\n')

            fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(10, 15))

            fig1.suptitle(f'ALM Solver Metrics, max iterations={params["maxInnerIter"]} T={T}', fontsize=16)


            color = 'tab:blue'
            ax1.set_ylabel('Cost', color=color)
            ax1.plot(self.ALM.costs[:], color=color, linestyle='-')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlabel('Iteration')  # Set the x-axis label
            ax1.grid(True)

            color = 'tab:red'
            ax2.set_ylabel('gradientNorms of Augmented Lagrangian(log10)', color=color)
            ax2.plot(np.log10(self.ALM.gradientNorms[:]), color=color, linestyle='-')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_xlabel('Iteration')  # Set the x-axis label
            ax2.grid(True)

            color = 'tab:red'
            ax3.set_ylabel('KKT(log10)', color=color)
            ax3.plot(np.log10(self.ALM.KKTs[:]), color=color, linestyle='-')
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.set_xlabel('Iteration')  # Set the x-axis label
            ax3.grid(True)

            color = 'tab:red'
            ax4.set_ylabel('Constraint violations(log10)', color=color)
            ax4.plot(np.log10(self.ALM.Infeasibilities[:]), color=color, linestyle='-')
            ax4.tick_params(axis='y', labelcolor=color)
            ax4.set_xlabel('Iteration')  # Set the x-axis label
            ax4.grid(True)
        
        elif solver == "ADMM":
            self.ADMM = SolverADMM(self.problem)

            start_time = time.time()
            self.ADMM.solve(init_xs, init_us)
            end_time = time.time()
            print(f'optimal control form ADMM solver: cost= {self.ADMM.cost},'
            f'\nus[0]:{self.ADMM.us[0]}, xs[0]:{self.ADMM.xs[0]}\nus[-1]:{self.ADMM.us[-1]}, xs[-1]:{self.ADMM.xs[-1]}\n\n')

            fig1, (ax1, ax3, ax4, ax5) = plt.subplots(4, sharex=True, figsize=(10, 15))

            fig1.suptitle(f'ADMM Solver Metrics  T={T}', fontsize=16)


            color = 'tab:blue'
            ax1.set_ylabel('Cost', color=color)
            ax1.plot(self.ADMM.costs[:], color=color, linestyle='-')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlabel('Iteration')  # Set the x-axis label
            ax1.grid(True)

            # color = 'tab:red'
            # ax2.set_ylabel('gradientNorms of Augmented Lagrangian(log10)', color=color)
            # ax2.plot(np.log10(self.ADMM.gradientNorms[:]), color=color, linestyle='-')
            # ax2.tick_params(axis='y', labelcolor=color)
            # ax2.set_xlabel('Iteration')  # Set the x-axis label
            # ax2.grid(True)

            color = 'tab:red'
            ax3.set_ylabel('KKT(log10)', color=color)
            ax3.plot(np.log10(self.ADMM.KKTs[:]), color=color, linestyle='-')
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.set_xlabel('Iteration')  # Set the x-axis label
            ax3.grid(True)

            color = 'tab:red'
            ax4.set_ylabel('Constraint violations(log10)', color=color)
            ax4.plot(np.log10(self.ADMM.Infeasibilities[:]), color=color, linestyle='-')
            ax4.tick_params(axis='y', labelcolor=color)
            ax4.set_xlabel('Iteration')  # Set the x-axis label
            ax4.grid(True)

            color = 'tab:red'
            ax5.set_ylabel('control variable magnitude', color=color)
            ax5.plot(self.ADMM.u_magnitude[:], color=color, linestyle='-')
            ax5.tick_params(axis='y', labelcolor=color)
            ax5.set_xlabel('Iteration')  # Set the x-axis label
            ax5.grid(True)
            
        elif solver == "MS":
            MS = SolverMS(self.problem)
            start_time = time.time()
            MS.solve(init_xs, init_us, params["maxIter"], params["alpha"])
            end_time = time.time()
            print(f'optimal control form multiple shooting solver: cost= {MS.cost},'
        f'\nus[0]:{MS.us[0]}, xs[0]:{MS.xs[0]}\nus[-1]:{MS.us[-1]}, xs[-1]:{MS.xs[-1]}\n\n')
            
            fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(10, 15))


            fig1.suptitle(f'MS Solver Metrics, max iterations={params["maxIter"]}, alpha = {params["alpha"]}, T={T}', fontsize=16)


            color = 'tab:blue'
            ax1.set_ylabel('Cost', color=color)
            ax1.plot(MS.costs[:], color=color, linestyle='-')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlabel('Iteration')  # Set the x-axis label
            ax1.grid(True)

            color = 'tab:red'
            ax2.set_ylabel('Constraint violations', color=color)
            ax2.plot(np.log10(MS.Infeasibilities[:]), color=color, linestyle='-')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_xlabel('Iteration')  # Set the x-axis label
            ax2.grid(True)

            color = 'tab:red'
            ax3.set_ylabel('KKT', color=color)
            ax3.plot(np.log10(MS.KKTs[:]), color=color, linestyle='-')
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.set_xlabel('Iteration')  # Set the x-axis label
            ax3.grid(True)

            color = 'tab:red'
            ax4.set_ylabel('Updating step Norm', color=color)
            ax4.plot(np.log10(MS.step_norm[:]), color=color, linestyle='-')
            ax4.tick_params(axis='y', labelcolor=color)
            ax4.set_xlabel('Iteration')  # Set the x-axis label
            ax4.grid(True)


            


        elif solver == "MSls":
            MSls = SolverMSls(self.problem)
            self.MSls = MSls
            start_time = time.time()
            MSls.solve(init_xs, init_us, params["maxIter"])
            end_time = time.time()
            print(f'optimal control form multiple shooting solver with line search: cost= {MSls.costs[-1]},'
        f'\nus[0]:{MSls.us[0]}, xs[0]:{MSls.xs[0]}\nus[-1]:{MSls.us[-1]}, xs[-1]:{MSls.xs[-1]}\n\n')
            
            fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, figsize=(10, 15))


            fig1.suptitle(f'MS Solver with line search Metrics, max iterations={params["maxIter"]}, T={T}', fontsize=16)


            color = 'tab:blue'
            ax1.set_ylabel('Cost (log10)', color=color)
            ax1.plot(MSls.costs[:], color=color, linestyle='-')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlabel('Iteration')  # Set the x-axis label
            ax1.grid(True)

            color = 'tab:red'
            ax3.set_ylabel('KKT (log10)', color=color)
            ax3.plot(np.log10(MSls.KKTs[:]), color=color, linestyle='-')
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.set_xlabel('Iteration')  # Set the x-axis label
            ax3.grid(True)


            color = 'tab:red'
            ax2.set_ylabel('Constraint violations (log10)', color=color)
            ax2.plot(np.log10(MSls.Infeasibilities[:]), color=color, linestyle='-')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_xlabel('Iteration')  # Set the x-axis label
            ax2.grid(True)

            color = 'tab:red'
            ax4.set_ylabel('alphas', color=color)
            ax4.plot(MSls.alphas[:], color=color, linestyle='-')
            ax4.tick_params(axis='y', labelcolor=color)
            ax4.set_xlabel('Iteration')  # Set the x-axis label
            ax4.grid(True)

            # color = 'tab:red'
            # ax5.set_ylabel('Updating step Norm', color=color)
            # ax5.plot(np.log10(MSls.step_norm[:]), color=color, linestyle='-')
            # ax5.tick_params(axis='y', labelcolor=color)
            # ax5.set_xlabel('Iteration')  # Set the x-axis label
            # ax5.grid(True)

            # Define the directory and filename

            folder_name = 'experiment/kuka_reaching_ocp'

            file_name = 'multiple_shooting.png'



            # Check if the folder exists, if not, create it

            if not os.path.exists(folder_name):

                os.makedirs(folder_name)



            # Full path

            full_path = os.path.join(folder_name, file_name)



            # Save the plot to the desired folder

            plt.savefig(full_path)


        elif solver == "hybrid":
            MSls = SolverMSls(self.problem)
            self.MSls = MSls
            MSls.solve(init_xs, init_us, params["maxIter"])

            us_warmStarted = MSls.us
            xs_warmStarted = MSls.xs

            
            DDP1 = crocoddyl.SolverFDDP(self.problem)
            self.FDDP_hybrid = DDP1
            DDP1.setCallbacks(
            [
                crocoddyl.CallbackLogger(),
                crocoddyl.CallbackVerbose(),
            ]
            )
            DDP1.solve(xs_warmStarted, us_warmStarted, 1000, False, 1e-3)

            print(f'optimal control form multiple shooting solver with line search: cost= {MSls.costs[-1]},'
        f'\nus[0]:{MSls.us[0]}, xs[0]:{MSls.xs[0]}\nus[-1]:{MSls.us[-1]}, xs[-1]:{MSls.xs[-1]}\n\n')
            
            print(f'optimal control form FDDP solver: cost= {DDP1.cost},'
        f'\nus[0]:{DDP1.us[0]}, xs[0]:{DDP1.xs[0]}\nus[-1]:{DDP1.us[-1]}, xs[-1]:{DDP1.xs[-1]}\n\n')
            
            fig2, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, figsize=(15, 20))


            fig2.suptitle(f'MS Solver with line search Metrics, max iterations={params["maxIter"]}, T={T}', fontsize=16)


            color = 'tab:blue'
            ax1.set_ylabel('Cost', color=color)
            ax1.plot(MSls.costs[:], color=color, linestyle='-')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.set_xlabel('Iteration')  # Set the x-axis label
            ax1.grid(True)

            color = 'tab:red'
            ax3.set_ylabel('KKT (log10)', color=color)
            ax3.plot(np.log10(MSls.KKTs[:]), color=color, linestyle='-')
            ax3.tick_params(axis='y', labelcolor=color)
            ax3.set_xlabel('Iteration')  # Set the x-axis label
            ax3.grid(True)

            color = 'tab:red'
            ax2.set_ylabel('Constraint violations (log10)', color=color)
            ax2.plot(np.log10(MSls.Infeasibilities[:]), color=color, linestyle='-')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_xlabel('Iteration')  # Set the x-axis label
            ax2.grid(True)

            color = 'tab:red'
            ax4.set_ylabel('Curvatures (log10)', color=color)
            ax4.plot(np.log10(abs(np.array(MSls.curvatures[:]))), color=color, linestyle='-')
            ax4.tick_params(axis='y', labelcolor=color)
            ax4.set_xlabel('Iteration')  # Set the x-axis label
            ax4.grid(True)

            color = 'tab:red'
            ax5.set_ylabel('Updating step Norm', color=color)
            ax5.plot(MSls.step_norm[:], color=color, linestyle='-')
            ax5.tick_params(axis='y', labelcolor=color)
            ax5.set_xlabel('Iteration')  # Set the x-axis label
            ax5.grid(True)

            color = 'tab:red'
            ax6.set_ylabel('Control magnitude', color=color)
            ax6.plot(MSls.u_magnitude[:], color=color, linestyle='-')
            ax6.tick_params(axis='y', labelcolor=color)
            ax6.set_xlabel('Iteration')  # Set the x-axis label
            ax6.grid(True)

            # log = DDP1.getCallbacks()[0]
            # # crocoddyl.plotOCSolution(log.xs, log.us)
            # crocoddyl.plotConvergence(
            #     log.costs, log.u_regs, log.x_regs, log.grads, log.stops, log.steps
            # )

        

if __name__ == '__main__':
    NX=CONFIG["NX"]; NU=CONFIG["NU"]; T=CONFIG["T"]; maxOuterIter=CONFIG["maxOuterIterations"]; maxInnerIter=CONFIG["maxInnerIterations"]
    T = 64
    tester = Tester(NX, NU, T)

    tester.problem = tester.problem_arm_manipulation

    # pdb.set_trace()

    print(f'problem: {tester.problem}')

    params_ddp = {
        "maxIter": 50
    }
    
    tester.test("FDDP", params_ddp)

    # params_MS = {
    #     "maxIter": 100,
    #     "alpha": 0.005
    # }

    # tester.test("MS", params_MS)

    params_MSls = {
        "maxIter": 40
    }


    
    tester.test("MSls", params_MSls)

    # tester.test("ALM", params_MSls)
    # print(f'Average control from FDDP = {np.linalg.norm(np.array(tester.FDDP_hybrid.us), 2)/T}\nAverage control from MSls = {np.linalg.norm(np.array(tester.MSls.us), 2)/T}')
    # print(f'Solution difference Average (warmstart) = {np.linalg.norm(np.array(tester.FDDP_hybrid.us)-np.array(tester.MSls.us), 2)/T}')
    # print(f'Solution difference Average (initial) = {np.linalg.norm(np.array(tester.FDDP_hybrid.us)-np.array(tester.init_us), 2)/T}')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    

