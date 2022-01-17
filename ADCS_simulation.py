#!/usr/bin/env python

import numpy as np
import math_utils
from phys_utils import ControlSystem
import plots
import pandas as pd
import argparse
import importlib.util
# import anomaly
# from tqdm import tqdm


def run_sim(simulation_settings):


    cs = ControlSystem(simulation_settings)

    simulation_time = 2  # hours

    loop_count = int(simulation_time * 3600 / cs.spacecraft.ctrl.T_loop)
    time_sec = np.linspace(0, (loop_count - 1) * cs.spacecraft.ctrl.T_loop, loop_count).T
    time_hours = time_sec / 3600.

    clock = np.zeros(loop_count)   # time vector
    x = np.zeros((7, loop_count))  # state vector
    x_est = np.zeros((cs.state_est_sz, loop_count))  # estimate of the state vector
    ang_rate_r = np.zeros((3, loop_count))

    clock[0] = 0.

    x[:4, 0] = math_utils.normalize(np.random.rand(4))
    x[4:7, 0] = np.random.rand(3) / 100.
    if simulation_settings['do_triad']:
        quat = cs.spacecraft.triad(clock[0], x[:4, 0])

        if quat is None:  # it might be best to propagate until triad hits on a solution or switch to else clause
            raise ValueError("TRIAD did not return attitude  quaternion")
        else:
            x_est[:4, 0] = quat
            x_est[4:7, 0] = cs.spacecraft.gyro.gyro_readings(x[4:7, 0])
    else:
        err_angle = 0.5 * np.random.rand(1) * 90. / 180.
        x_err = math_utils.normalize(np.concatenate((np.cos(err_angle), np.random.rand(3) * np.sin(err_angle)), 0))
        x_est[:4, 0] = math_utils.quat_product(x[:4, 0], x_err)
        x_est[4:7, 0] = np.random.rand(3) / 100.

    for i in range(1, loop_count):
        print(i, loop_count)
        clock[i], x[:, i], x_est[:, i], ang_rate_r[:, i], m_ctrl, telemetry = \
            cs.control_loop(clock[i-1], x[:, i-1], x_est[:, i-1])

    return time_hours, clock, x, x_est, ang_rate_r, m_ctrl, telemetry


def load_settings(filename):
    spec = importlib.util.spec_from_file_location('sim_settings', filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.simulation_settings

def main():
    parser = argparse.ArgumentParser(description="A program for generating state and telemetry data for a cubesat")
    parser.add_argument('sim', type=str, help="Name of simulation. Simulation configuration is assumed to be in sim.conf.py")
    parser.add_argument('--res', metavar='result_path', type=str, default=None, help="A path to place results. Do not save results if not specified")
    parser.add_argument('--plots', action='store_true', help='Builds plots if present')
    args = parser.parse_args()

    simulation_settings = load_settings(args.sim)
    time_hours, clock, x, x_est, ang_rate_r, m_ctrl, telemetry = run_sim(simulation_settings)

    t = pd.DataFrame(clock, columns=["time"])
    quat = pd.DataFrame(x[:4, :].T, columns=["q1", "q2", "q3", "q4"])
    ang_rate_a = pd.DataFrame(x[4:7, :].T, columns=["omg1", "omg2", "omg3"])
    ang_rate_r = pd.DataFrame(ang_rate_r.T, columns=["Omgr1", "Omgr2", "Omgr3"])
    quat_est = pd.DataFrame(x_est[:4, :].T, columns=["q_est1", "q_est2", "q_est3", "q_est4"])
    ang_rate_a_est = pd.DataFrame(x_est[4:7, :].T, columns=["omg1_est", "omg2_est", "omg3_est"])
    df = pd.concat((t, quat, ang_rate_a, ang_rate_r, quat_est, ang_rate_a_est), axis=1)

    if args.plots:
        plots.DynamicsPlot(time_hours, x[:, 0], x, x_est, df[["Omgr1", "Omgr2", "Omgr3"]].to_numpy().T)

    if args.res:
        df.to_csv(args.res)


if __name__ == '__main__':
    main()

