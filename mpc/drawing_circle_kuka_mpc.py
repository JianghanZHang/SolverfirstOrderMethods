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
from solverNAG import SolverNAG
from solverADAM import SolverADAM
import time

np.set_printoptions(precision=4, linewidth=180)

def solveOCP(solver, x_curr, us_prev, targets, maxIter, alpha = None):
    solver.problem.x0 = x_curr
    us_init = list(us_prev[1:]) + [us_prev[-1]]
    xs_init = list(solver.xs[1:]) + [solver.xs[-1]]
    xs_init[0] = x_curr

    # Get OCP nodes
    models = list(solver.problem.runningModels) + [solver.problem.terminalModel]
    for k, model in enumerate(models):
        model.differential.costs.costs["translation"].active = True
        model.differential.costs.costs["translation"].cost.residual.reference = targets[k]
        model.differential.costs.costs["translation"].weight = 1e2

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

    env = BulletEnvWithGround(p.GUI, dt=1e-3)
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
    runningCostModel.addCost("translation", frameTranslationCost, 1e2)

    terminalCostModel.addCost("stateReg", xRegCost, 1e-1)
    terminalCostModel.addCost("translation", frameTranslationCost, 1e2)

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

    time_ = 2.
    t = 0.
    dt_sim = env.dt  # 1e-3
    sim_freq = 1/dt_sim
    dt_mpc = 1e-3
    mpc_freq = 1/dt_mpc
    num_step = int(time_ / dt_sim)

    # warm starting us
    x_measured = x0
    xs_init = [x0 for i in range(T + 1)]
    us_init = ddp.problem.quasiStatic(xs_init[:-1])
    ddp.solve(xs_init, us_init, False)
    us = np.array(ddp.us)
    xs = np.array(ddp.xs)

    mpc_utils.display_ball(ee_translation, RADIUS=.05, COLOR=[1., 0., 0., .6])
    q_measured = q0

    totalCosts = []
    runningCosts = []
    KKTs = []
    ee_positions = []
    # Simulating with MPC freq == simulation freq
    maxIter = 10
    log_rate = 100
    alpha = 1
    for i in range(num_step):

        tau_gravity = pin.rnea(pin_robot.model, pin_robot.data, q_measured, np.zeros_like(q_measured), np.zeros_like(q_measured))

        if i % (int(sim_freq / mpc_freq)) == 0:
            targets = circleTraj(T, t, dt_ocp)
            us, xs, runningCost, totalCost, kkt = solveOCP(adam, x_measured, us, targets, maxIter, alpha)
            runningCosts.append(runningCost)
            totalCosts.append(totalCost)
            KKTs.append(kkt)
            tau = us[0]

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

    # Set the figure size
    fig1, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, sharex=True, figsize=(10, 15))

    fig1.suptitle(f'ADAM online Metrics, Horizon Length={T}, maxIter={maxIter}, lr={alpha}', fontsize=16)

    color = 'tab:blue'
    ax1.set_ylabel('runningCost', color=color)
    ax1.plot(totalCosts[:], color=color, linestyle='-')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xlabel('Iteration')  # Set the x-axis label
    ax1.grid(True)

    color = 'tab:red'
    ax2.set_ylabel('totalCost of OCP', color=color)
    ax2.plot(totalCosts[:], color=color, linestyle='-')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_xlabel('Iteration')  # Set the x-axis label
    ax2.grid(True)

    color = 'tab:green'
    ax3.set_ylabel('KKT(log10)', color=color)
    ax3.plot(np.log10(KKTs[:]), color=color, linestyle='-')
    ax3.tick_params(axis='y', labelcolor=color)
    ax3.set_xlabel('Iteration')  # Set the x-axis label
    ax3.grid(True)

    color = 'tab:blue'
    ax4.set_ylabel('ee_position_x', color=color)
    ax4.plot(ee_positions[:, 0], color=color, linestyle='--')
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.set_xlabel('Iteration')  # Set the x-axis label
    ax4.grid(True)

    color = 'tab:red'
    ax5.set_ylabel('ee_position_y', color=color)
    ax5.plot(ee_positions[:, 1], color=color, linestyle='--')
    ax5.tick_params(axis='y', labelcolor=color)
    ax5.set_xlabel('Iteration')  # Set the x-axis label
    ax5.grid(True)

    color = 'tab:green'
    ax6.set_ylabel('ee_position_z', color=color)
    ax6.plot(ee_positions[:, 2], color=color, linestyle='--')
    ax6.tick_params(axis='y', labelcolor=color)
    ax6.set_xlabel('Iteration')  # Set the x-axis label
    ax6.grid(True)

    plt.savefig(f'plots/online/ADAM_online_noGravComp2.png')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

