import pdb
import pinocchio as pin
import crocoddyl
import numpy as np
from bullet_utils.env import BulletEnvWithGround
from robot_properties_kuka.iiwaWrapper import IiwaRobot
import pybullet as p
import mpc_utils
from solverILQR import SolverILqr
from solverLBGFS_vectorized import SolverLBGFS
from solverGD import SolverGD

import matplotlib.pyplot as plt
from solverNAG_lineSearch import SolverNAG
from solverADAM_lineSearch import SolverADAM
from solverAMSGRAD import SolverAMSGRAD
from solverNADAM import SolverNADAM
from solverADAN_lineSearch import SolverADAN

import time

np.set_printoptions(precision=4, linewidth=180)

def solveOCP(solver, x_curr, us_prev, targets, maxIter, alpha=None):
    solver.problem.x0 = x_curr

    us_init = list(us_prev[1:]) + [us_prev[-1]]
    xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
    xs_init[0] = x_curr

    # Get OCP nodes
    models = list(solver.problem.runningModels) + [solver.problem.terminalModel]
    for k, model in enumerate(models):
        model.differential.costs.costs["translation"].active = True
        model.differential.costs.costs["translation"].cost.residual.reference = targets[k]
        model.differential.costs.costs["translation"].weight = 100

    if alpha is None:
        solver.solve(xs_init, us_init, maxIter, False)
    else:
        solver.solve(xs_init, us_init, maxIter, False, alpha)
    # calculating cost of the current node
    u_curr = solver.us[0]
    runningModel0 = solver.problem.runningModels[0]
    runningData0 = solver.problem.runningDatas[0]
    runningModel0.calc(runningData0, x_curr, u_curr)
    runningCost = runningData0.cost

    totalCost = solver.cost

    Qu = solver.Qu
    kkt = np.linalg.norm(Qu, 2)

    return np.array(solver.us), np.array(solver.xs), runningCost, totalCost, kkt

def circleTraj(T, t, dt):
    pi = np.pi
    target = np.zeros([T + 1, 3])
    for j in range(T + 1):
        target[j, 0] = .4 + (.1 * np.cos(pi * (t + j*dt)))
        target[j, 1] = .2 + (.1 * np.sin(pi * (t + j*dt)))
        target[j, 2] = .3

    return target


if __name__ == '__main__':

    env = BulletEnvWithGround(p.GUI, dt=1e-2)
    robot_simulator = IiwaRobot()
    pin_robot = robot_simulator.pin_robot
    q0 = np.array([0.9755, 1.2615, 1.7282, 1.8473, -1.0791, 2.0306, -0.0759])
    v0 = np.zeros(pin_robot.model.nv)
    x0 = np.concatenate([q0, v0])
    env.add_robot(robot_simulator)
    robot_simulator.reset_state(q0, v0)
    robot_simulator.forward_robot(q0, v0)

    state = crocoddyl.StateMultibody(pin_robot.model)
    actuation = crocoddyl.ActuationModelFull(state)
    runningCostModel = crocoddyl.CostModelSum(state)
    terminalCostModel = crocoddyl.CostModelSum(state)
    ee_frame_id = robot_simulator.pin_robot.model.getFrameId("contact")
    ee_translation = np.array([.4, .2, .3])

    frameTranslationResidual = crocoddyl.ResidualModelFrameTranslation(state, ee_frame_id, ee_translation)
    frameTranslationCost = crocoddyl.CostModelResidual(state, frameTranslationResidual)
    uResidual = crocoddyl.ResidualModelControlGrav(state)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    xResidual = crocoddyl.ResidualModelState(state, x0)
    xRegCost = crocoddyl.CostModelResidual(state, xResidual)
    runningCostModel.addCost("stateReg", xRegCost, 1e-1)
    runningCostModel.addCost("ctrlRegGrav", uRegCost, 1e-4)
    runningCostModel.addCost("translation", frameTranslationCost, 100)

    terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
    terminalCostModel.addCost("translation", frameTranslationCost, 100)

    running_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, runningCostModel)
    terminal_DAM = crocoddyl.DifferentialActionModelFreeFwdDynamics(state, actuation, terminalCostModel)
    dt_ocp = 1e-2
    runningModel = crocoddyl.IntegratedActionModelEuler(running_DAM, dt_ocp)
    terminalModel = crocoddyl.IntegratedActionModelEuler(terminal_DAM, 0.)
    T = 30
    problem = crocoddyl.ShootingProblem(x0, [runningModel] * T, terminalModel)

    ddp = crocoddyl.SolverDDP(problem)

    ilqr = SolverILqr(problem)
    lbfgs = SolverLBGFS(problem, memory_length=30)
    gd = SolverGD(problem)
    nag = SolverNAG(problem)
    adam = SolverADAM(problem)
    NAdam = SolverNADAM(problem)
    amsGrad = SolverAMSGRAD(problem)
    adan = SolverADAN(problem)

    time_ = 2.
    t = 0.
    dt_sim = env.dt  # 1e-3
    sim_freq = 1/dt_sim
    dt_mpc = env.dt
    mpc_freq = 1/dt_mpc
    num_step = int(time_ / dt_sim)

    # warm starting us
    x_measured = x0
    xs_init = [x0 for i in range(T + 1)]
    us_init = ddp.problem.quasiStatic(xs_init[:-1])
    targets = circleTraj(T, t, dt_ocp)
    us, xs, runningCost, totalCost, kkt = solveOCP(ddp, x0, us_init, targets, 1000)
    us = np.array(us)
    xs = np.array(xs)

    mpc_utils.display_ball(ee_translation, RADIUS=.05, COLOR=[1., 0., 0., .6])
    q_measured = q0

    totalCosts = []
    runningCosts = []
    KKTs = []
    ee_positions = []
    cost_examples = []
    kkt_examples = []
    update_examples = []
    curvature_examples = []
    alpha_examples = []

    log_rate = 100
    #alpha = 1.
    solver = adam
    solver.Beta1 = .9
    solver.Beta2 = .999
    # solver.Beta3 = .999
    # solver.decay1 = 1.
    # solver.decay2 = 1.
    # solver.decay3 = 1.
    solver.bias_correction = True
    solver.refresh = False
    for i in range(num_step):

        tau_gravity = pin.rnea(pin_robot.model, pin_robot.data, q_measured, np.zeros_like(q_measured), np.zeros_like(q_measured))

        if i % (int(sim_freq / mpc_freq)) == 0:
            targets = circleTraj(T, t, dt_ocp)
            maxIter = 10
            us, xs, runningCost, totalCost, kkt = solveOCP(solver, x_measured, us, targets, maxIter)#, alpha)
            runningCosts.append(runningCost)
            totalCosts.append(totalCost)
            KKTs.append(kkt)
            tau = us[0]

            # if i in range(0, 200, 20):
            #     cost_examples.append(solver.costs)
            #     kkt_examples.append(solver.KKTs)
            #     update_examples.append(solver.updates)
            #     curvature_examples.append(solver.curvatures)
            #     alpha_examples.append(solver.alphas)

            if i % log_rate == 0:
                print(f'at step {i}: tau={tau}')

            # tau += tau_gravity
            robot_simulator.send_joint_command(tau)

            ee_positions.append(np.array(robot_simulator.pin_robot.data.oMf[ee_frame_id].translation))

            env.step()
            q_measured, v_measured = robot_simulator.get_state()
            robot_simulator.forward_robot(q_measured, v_measured)
            x_measured = np.concatenate([q_measured, v_measured])
            t += dt_sim

    ee_positions = np.array(ee_positions)

    # for t, (cost_example, kkt_example, update_example, curvature_example, alpha_example) \
    #         in enumerate(zip(cost_examples, kkt_examples, update_examples, curvature_examples, alpha_examples)):
    #
    #     fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(10, 15))
    #
    #     fig.suptitle(f'in {t} example', fontsize=16)
    #
    #     color = 'tab:blue'
    #     ax1.set_ylabel('Costs', color=color)
    #     ax1.plot(cost_example[:], color=color, linestyle='-')
    #     ax1.tick_params(axis='y', labelcolor=color)
    #     ax1.set_xlabel('Iteration')  # Set the x-axis label
    #     ax1.grid(True)
    #
    #     color = 'tab:red'
    #     ax2.set_ylabel('L2 norm of gradients', color=color)
    #     ax2.plot(kkt_example[:], color=color, linestyle='-')
    #     ax2.tick_params(axis='y', labelcolor=color)
    #     ax2.set_xlabel('Iteration')  # Set the x-axis label
    #     ax2.grid(True)
    #
    #     color = 'tab:green'
    #     ax3.set_ylabel('L2 norm of updating vector', color=color)
    #     ax3.plot(update_example[:], color=color, linestyle='-')
    #     ax3.tick_params(axis='y', labelcolor=color)
    #     ax3.set_xlabel('Iteration')  # Set the x-axis label
    #     ax3.grid(True)
    #
    #     color = 'tab:blue'
    #     ax4.set_ylabel('Curvature', color=color)
    #     ax4.plot(curvature_example[:], color=color, linestyle='-')
    #     ax4.tick_params(axis='y', labelcolor=color)
    #     ax4.set_xlabel('Iteration')  # Set the x-axis label
    #     ax4.grid(True)
    #
    #     color = 'tab:blue'
    #     ax5.set_ylabel('Alphas', color=color)
    #     ax5.plot(alpha_example[:], color=color, linestyle='-')
    #     ax5.tick_params(axis='y', labelcolor=color)
    #     ax5.set_xlabel('Iteration')  # Set the x-axis label
    #     ax5.grid(True)
    #


    # Set the figure size
    fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, figsize=(10, 15))

    fig1.suptitle(f'online Metrics, Sovler={solver}, '
                  f'Betas={solver.Beta1, solver.Beta2}\n Max_iteration={maxIter},'
                  f'Bias_correction = {solver.bias_correction}, Refresh_moment = {solver.refresh}', fontsize=16)

    start = 0
    color = 'tab:blue'
    ax1.set_ylabel('runningCost', color=color)
    ax1.plot(runningCosts[start:], color=color, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    color = 'tab:red'
    ax2.set_ylabel('totalCosts', color=color)
    ax2.plot(totalCosts[start:], color=color, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(True)

    color = 'tab:green'
    ax3.set_ylabel('KKT(log10)', color=color)
    ax3.plot(np.log10(KKTs[start:]), color=color, linestyle='-')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.grid(True)

    color = 'tab:blue'
    ax4.set_ylabel('ee_position_x', color=color)
    ax4.plot(ee_positions[start:, 0], color=color, linestyle='--')
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.grid(True)

    color = 'tab:red'
    ax5.set_ylabel('ee_position_y', color=color)
    ax5.plot(ee_positions[start:, 1], color=color, linestyle='--')
    ax5.tick_params(axis='y', labelcolor=color)
    ax5.grid(True)

    color = 'tab:green'
    ax6.set_ylabel('ee_position_z', color=color)
    ax6.plot(ee_positions[start:, 2], color=color, linestyle='--')
    ax6.tick_params(axis='y', labelcolor=color)
    ax6.set_xlabel('time step')  # Set the x-axis label
    ax6.grid(True)

    plt.savefig(f'plots/online/ADAM_online_withLineSearch0.png')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()
    print(f'line search fail: {solver.fail_ls}')
    print(f'guess accepted: {sum(solver.guess_accepted)}')
    print(f'line search failed: {sum(solver.lineSearch_fail)}')

