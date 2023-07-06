import os
import signal
import sys
import time

import example_robot_data
import numpy as np
import pinocchio

import crocoddyl


def quadrotor_problem():
    WITHDISPLAY = "display" in sys.argv or "CROCODDYL_DISPLAY" in os.environ
    WITHPLOT = "plot" in sys.argv or "CROCODDYL_PLOT" in os.environ
    signal.signal(signal.SIGINT, signal.SIG_DFL)

    hector = example_robot_data.load("hector")
    robot_model = hector.model

    target_pos = np.array([1.0, 0.0, 1.0])
    target_quat = pinocchio.Quaternion(1.0, 0.0, 0.0, 0.0)

    state = crocoddyl.StateMultibody(robot_model)

    d_cog, cf, cm, u_lim, l_lim = 0.1525, 6.6e-5, 1e-6, 5.0, 0.1
    tau_f = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, d_cog, 0.0, -d_cog],
            [-d_cog, 0.0, d_cog, 0.0],
            [-cm / cf, cm / cf, -cm / cf, cm / cf],
        ]
    )
    actuation = crocoddyl.ActuationModelMultiCopterBase(state, tau_f)

    nu = actuation.nu
    runningCostModel = crocoddyl.CostModelSum(state, nu)
    terminalCostModel = crocoddyl.CostModelSum(state, nu)

    # Costs
    xResidual = crocoddyl.ResidualModelState(state, state.zero(), nu)
    xActivation = crocoddyl.ActivationModelWeightedQuad(
        np.array([0.1] * 3 + [1000.0] * 3 + [1000.0] * robot_model.nv)
    )
    uResidual = crocoddyl.ResidualModelControl(state, nu)
    xRegCost = crocoddyl.CostModelResidual(state, xActivation, xResidual)
    uRegCost = crocoddyl.CostModelResidual(state, uResidual)
    goalTrackingResidual = crocoddyl.ResidualModelFramePlacement(
        state,
        robot_model.getFrameId("base_link"),
        pinocchio.SE3(target_quat.matrix(), target_pos),
        nu,
    )
    goalTrackingCost = crocoddyl.CostModelResidual(state, goalTrackingResidual)
    runningCostModel.addCost("xReg", xRegCost, 1e-6)
    runningCostModel.addCost("uReg", uRegCost, 1e-6)
    runningCostModel.addCost("trackPose", goalTrackingCost, 1e-2)
    terminalCostModel.addCost("goalPose", goalTrackingCost, 3.0)

    dt = 3e-2
    runningModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, runningCostModel
        ),
        dt,
    )
    terminalModel = crocoddyl.IntegratedActionModelEuler(
        crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, terminalCostModel
        ),
        dt,
    )
    # runningModel.u_lb = np.array([l_lim, l_lim, l_lim, l_lim])
    # runningModel.u_ub = np.array([u_lim, u_lim, u_lim, u_lim])

    # Creating the shooting problem and the BoxDDP solver
    T = 33
    problem = crocoddyl.ShootingProblem(
        np.concatenate([hector.q0, np.zeros(state.nv)]), [runningModel] * T, terminalModel
    )
    return problem