import numpy as np
simulation_settings = {'inertia_tensor': np.diag(np.array([0.014, 0.015, 0.007])),  # [kg*m^2] sc inertia tensor
                       'start_time': "2022-05-27 12:00",  # mission start date and time
                       'altitude': 409e+3,  # [m] altitude of the orbit
                       'inclination': 51.63 * np.pi / 180,  # [rad] inclination of the orbit (51.63 * np.pi / 180)
                       'ctrl_meas_time': 1.,           # [s] sampling time step
                       'ctrl_ctrl_time': 6.,           # [s] control time step
                       'env_trq_sigma': 5e-9,         # [N * m] disturbance of the torque
                       'env_magn_field_sigma': 1e-7,  # [T] noise to create actual magnetic field
                       'do_filter': True,             # boolean use Kalman filter
                       'do_triad':  False,             # boolean, use TRIAD to initialize Kalman filter
                       'use_magnetometer': True,      # boolean
                       'magnetometer_bias': 1e-8,     # [T] magnetometer bias sigma (to generate 3d random const bias)
                       'magnetometer_noise': 1e-7,    # [T] magnetometer noise (sigma)
                       'magnetometer_pos': np.array([0.03, 0.035, 0.04]),  # [m] magnetometer position
                       'use_gyro': True,             # boolean
                       'gyro_bias': 1e-5,             # [rad/s] gyroscope bias sigma (to generate 3d random const bias)
                       'gyro_noise': 1e-4,            # [rad/s] gyroscope noise
                       'use_sun_sensors': False,      # boolean
                       'sun_sensors_axes': np.array([[1., 0., 0.],
                                                     [0., 1., 0.],
                                                     [0., 0., 1.],
                                                     [-1., 0., 0.],
                                                     [0., -1., 0.]]),  # fov axes of sun sensors
                       'sun_sensor_fov': np.pi / 3,   # [rad] half con of the field of view
                       'sun_sensor_noise': 5e-3,      # [rad] 1 sigma for Sun direction error
                       'const_m_res': 3e-5,           # [Am^2] constant residual moment absolute value
                       'ekf_Q_sigma_m_res': 1e-7,     # residual moment uncertainty for process noise
                       'ekf_Q_sigma_b_bias': 1e-6,    # magnetometer bias uncertainty for process noise
                       'ekf_Q_sigma_g_bias': 1e-6,    # gyro bias uncertainty for process noise
                       'ekf_P0_sigma_q0': np.pi / 2.,  # uncertainty in the initial attitude
                       'ekf_P0_sigma_omega0': 1e-2,    # uncertainty in the initial angular velocity
                       'ekf_P0_sigma_mres': 3e-2,      # uncertainty in the initial residual magnetic moment
                       'ekf_P0_sigma_mbias': 1e-5,     # uncertainty in the initial magnetometer bias
                       'ekf_P0_sigma_gbias': 1e-5,      # uncertainty in the initial gyro bias
                       'mtq_V_max': 7,
                       'mtq_coil_resistance': 25.,
                       'mtq_alpha': 0.,
                       'mtq_n_turns': 200.,
                       'mtq_coil_area': 0.05**2,
                       'mtq_noise_sigma': 1e-10,
                       'pd_Kw_coef': 60.,
                       'pd_Ka': 12.,
                       }
