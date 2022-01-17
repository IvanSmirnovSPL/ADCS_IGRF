import numpy as np
from scipy.integrate import solve_ivp
import math_utils
from enum import Enum
from astropy.time import Time
from astropy.coordinates import solar_system_ephemeris, EarthLocation
from astropy.coordinates import get_body_barycentric, get_body, get_moon
from datetime import timedelta, datetime
from IGRF import IGRF


class Parameters(object):
    pass


cnst = Parameters()
cnst.MU = 3.986e+14   # [m^3 / s^2] standard gravitational parameter of the Earth
cnst.MU_0 = 1.257e-6  # [N / A^2] vacuum permeability
cnst.MU_e = 7.94e+22  # [A * m^2] magnetic dipole moment of the Earth
cnst.R_e = 6371e+3    # [m] radius of the Earth
cnst.J2 = 0.00108263


class Magnetometer:
    def __init__(self, bias_sigma, noise_sigma, position=np.zeros(3), permutation_matrix=np.eye(3)):
        self.bias = np.random.normal(0, bias_sigma, 3)
        self.noise_sigma = noise_sigma
        self.pos = position
        self.A = permutation_matrix
        self.on = True

    def magnetometer_switch(self, on=True):
        self.on = on

    def magnetometer_readings(self, magnetic_field):
        return self.on * (magnetic_field + self.bias + np.random.normal(0, self.noise_sigma, 3))

    def magnetometer_readings_permuted(self, magnetic_field):
        return self.on * self.A.dot(magnetic_field + self.bias + np.random.normal(0, self.noise_sigma, 3))


class Gyroscope:
    def __init__(self, bias_sigma, noise_sigma, permutation_matrix=np.eye(3)):
        self.bias = np.random.normal(0, bias_sigma, 3)
        self.noise_sigma = noise_sigma
        self.A = permutation_matrix
        self.on = True

    def gyro_switch(self, on=True):
        self.on = on

    def gyro_readings(self, ang_rate):
        return self.on * (ang_rate + self.bias + np.random.normal(0, self.noise_sigma, 3))

    def gyro_readings_permuted(self, ang_rate):
        return self.on * self.A.dot(ang_rate + self.bias + np.random.normal(0, self.noise_sigma, 3))


class SunSensor:
    def __init__(self, mount_axis, fov_angle, noise_sigma):
        self.axis = mount_axis / np.linalg.norm(mount_axis)
        self.fov = fov_angle
        self.cos_fov = np.cos(self.fov)
        self.noise_sigma = noise_sigma
        self.on = True

    def ss_switch(self, on=True):
        self.on = on

    def ss_readings(self, e_sun, eclipsed=False):
        if eclipsed or e_sun.dot(self.axis) < self.cos_fov:
            intensity = 0.
            return np.zeros(3), intensity  # can be Moon or albedo
        else:
            intensity = self.on * e_sun.dot(self.axis)
            return self.on * (e_sun + np.random.normal(0, self.noise_sigma, 3)), intensity


class Magnetorquer:
    def __init__(self, mount_axis, Vmax, resistance, alpha, area, n_turns, noise_sigma):
        self.axis = mount_axis / np.linalg.norm(mount_axis)
        self.R = resistance
        self.alpha = alpha
        self.I2m = area * n_turns
        self.m_max = (Vmax / self.R) * self.I2m
        self.noise_sigma = noise_sigma
        self.on = True

    def mtq_switch(self, on=True):
        self.on = on

    def magnetic_moment(self, V_ctrl, inv=1):
        m_ctrl = inv * V_ctrl / (self.R * (1 + self.alpha)) * self.I2m

        return self.on * self.axis * (m_ctrl + np.random.normal(0, self.noise_sigma, 1))


class CircularOrbit:
    def __init__(self, altitude, inclination, raan=0.):
        self.altitude = altitude
        self.incl = inclination
        self.raan = raan
        self.R = cnst.R_e + self.altitude  # [m] radius of the orbit
        self.mean_motion = np.sqrt(cnst.MU / self.R**3.)  # [rad / s] angular velocity on the orbit
        self.V = self.mean_motion * self.R  # [m/s] velocity for the circular orbit

    def unit_position_velocity(self, arg_lat):
        raan_matr = np.array([[np.cos(self.raan), -np.sin(self.raan), 0.],
                              [np.sin(self.raan), np.cos(self.raan), 0.],
                              [0., 0., 1.]])
        incl_matr = np.array([[1., 0., 0.],
                              [0., np.cos(self.incl), -np.sin(self.incl)],
                              [0., np.sin(self.incl), np.cos(self.incl)]])
        lat_matr = np.array([[np.cos(arg_lat), -np.sin(arg_lat), 0.],
                             [np.sin(arg_lat), np.cos(arg_lat), 0.],
                             [0., 0., 1.]])

        e_pos = raan_matr.dot(incl_matr).dot(lat_matr).dot(np.array([1., 0., 0.]))
        e_vel = raan_matr.dot(incl_matr).dot(lat_matr).dot(np.array([0., 1., 0.]))

        return e_pos, e_vel

    def eci2orb(self, arg_lat):
        e_pos, e_vel = self.unit_position_velocity(arg_lat)

        return np.vstack((e_vel, np.cross(e_pos, e_vel), e_pos))


class Environment:
    def __init__(self, r, trq_sigma, magn_field_sigma, str_time, eci2orb):
        self.B_0 = cnst.MU_e * cnst.MU_0 / (4 * np.pi * r ** 3.)
        self.B_1 = cnst.MU_0 / (4 * np.pi)
        self.trq_sigma = trq_sigma
        self.magn_sigma = magn_field_sigma
        self.start_time = datetime.strptime(str_time[:-6], '%Y-%m-%d') \
                          + timedelta(hours=float(str_time[-5:-3]))
        self.altitude = r
        self.eci2orb = eci2orb

    def direct_dipole(self, u, incl):
        b = self.B_0 * np.array([np.cos(u) * np.sin(incl), np.cos(incl), -2. * np.sin(u) * np.sin(incl)])

        return b

    def direct_dipole_(self, u, incl, time):
        # print(u, incl, time)
        ct = self.start_time + timedelta(seconds=time)
        # current_time = get_time.get_current_time(current_time)
        TIME = IGRF.greg_to_julian((ct.year, ct.month, ct.day,
                                    ct.hour, ct.minute, ct.second))
        theta = incl % np.pi - np.pi / 2
        phi = u % (2 * np.pi) - np.pi
        B = IGRF.magn_field_ECI(TIME, self.altitude, theta, phi)
        B *= 1e-9
        #print('B', B)
        B = self.eci2orb(u).dot(B)
        return B


    def external_magnetic_field(self, b_ideal):

        return b_ideal + np.random.normal(0, self.magn_sigma, 3)

    def sun_direction_eci(self, t):
        #time = Time("1996-05-27 12:00")  # t
        time = Time(t)
        loc = EarthLocation.of_site('greenwich')
        solar_system_ephemeris.set('de430')
        loc_sun = get_body('sun', time, loc)
        r_sun = np.array([loc_sun.cartesian.x.value, loc_sun.cartesian.y.value, loc_sun.cartesian.z.value])
        e_sun = math_utils.normalize(r_sun)

        return r_sun, e_sun

    def sight(self, rA, rB):
        # adopted from D. Vallado
        # this function takes the position vectors of two satellites and determines
        # if there is line-of-sight between the two satellites.
        bsqrd = np.linalg.norm(rB)**2
        asqrd = np.linalg.norm(rA)**2
        adotb = rA.dot(rB)

        # find tmin
        if np.abs(asqrd + bsqrd - 2.0 * adotb) < 1e-4:
            dmin = 0.0
        else:
            dmin = (asqrd - adotb) / (asqrd + bsqrd - 2.0 * adotb)

        if (dmin < 0.) or (dmin > 1.0):
            return 1.
        else:
            return 1 < ((1.0 - dmin) * asqrd + adotb * dmin) / cnst.R_e**2

    def grav_torque(self, quat, mean_motion, inertia_tensor):
        Ae3 = math_utils.rotate_vec_with_quat(quat, np.array([0., 0., 1.]))

        return 3. * mean_motion ** 2. * math_utils.cross_product(Ae3, inertia_tensor.dot(Ae3))

    def magnetic_torque(self, magnetic_moment, magnetic_field):

        return math_utils.cross_product(magnetic_moment, magnetic_field)

    def dstrb_torque(self):

        return np.random.normal(0, self.trq_sigma)

    def external_torque(self, quat, mean_motion, inertia_tensor, dstrb=True):
        if dstrb:
            return self.grav_torque(quat, mean_motion, inertia_tensor) + self.dstrb_torque()
        else:
            return self.grav_torque(quat, mean_motion, inertia_tensor)


class FilterType(Enum):
    NO_FILTER = 0
    MTM = 1
    MTM_GYRO = 2
    MTM_GYRO_SUN = 3


def filter_type(cs_dict):
    if cs_dict['use_magnetometer'] and cs_dict['use_gyro'] and cs_dict['use_sun_sensors']:
        return FilterType.MTM_GYRO_SUN
    elif cs_dict['use_magnetometer'] and cs_dict['use_gyro']:
        return FilterType.MTM_GYRO
    elif cs_dict['use_magnetometer']:
        return FilterType.MTM
    else:
        return FilterType.NO_FILTER


class ExtendedKalmanFilter:
    def __init__(self, cs_dict):
        self.sc = Spacecraft(cs_dict)
        self.tau = self.sc.ctrl.T_loop
        self.sensors_used = filter_type(cs_dict)
        self.state_sz = self.state_size()
        # self.Phi0 = np.eye(12)
        # self.Phi0[3, 3] = self.Phi0[4, 4] = self.Phi0[5, 5] = 0.5 * self.tau
        self.Q0 = self.ini_process_covariance(cs_dict)
        self.P0 = self.ini_error_covariance(cs_dict)
        self.P_est = self.P0
        self.R_cov = self.ini_measurements_covariance(cs_dict)

    def state_size(self):
        if self.sensors_used == FilterType.MTM:
            return 13  # quaternion (4), ang_rate (3), res magnetic moment (3), magn bias (3)
        elif self.sensors_used in (FilterType.MTM_GYRO, FilterType.MTM_GYRO_SUN):
            return 16  # quaternion (4), ang_rate (3), res magnetic moment (3), magn bias (3), gyro bias (3)
        else:
            raise ValueError("Unidentified set of sensors")

    def ini_error_covariance(self, cs_dict):

        P0 = np.eye(self.state_sz - 1)
        row, col = np.diag_indices(P0.shape[0])

        P0[row[:3], col[:3]] = cs_dict['ekf_P0_sigma_q0']**2
        P0[row[3:6], col[3:6]] = cs_dict['ekf_P0_sigma_omega0']**2
        P0[row[6:9], col[6:9]] = cs_dict['ekf_P0_sigma_mres']**2
        P0[row[9:12], col[9:12]] = cs_dict['ekf_P0_sigma_mbias']**2

        if self.sensors_used in (FilterType.MTM_GYRO, FilterType.MTM_GYRO_SUN):
            P0[row[12:15], col[12:15]] = cs_dict['ekf_P0_sigma_gbias']**2

        return P0

    def ini_process_covariance(self, cs_dict):
        G_6x3 = np.concatenate((np.zeros((3, 3)), self.sc.J_inv), 0)
        G_6x9_top = np.concatenate((G_6x3, np.zeros((6, 6))), 1)
        G_6x9_bottom = np.concatenate((np.zeros((6, 3)), np.eye(6)), 1)
        G = np.concatenate((G_6x9_top, G_6x9_bottom), 0)

        if self.sensors_used == FilterType.MTM:
            D = np.eye(9)
        elif self.sensors_used in (FilterType.MTM_GYRO, FilterType.MTM_GYRO_SUN):
            D = np.eye(12)
            G = np.concatenate((G, np.zeros((12, 3))), 1)
            G = np.concatenate((G, np.zeros((3, 12))), 0)
            G[12:15, 9:12] = np.eye(3)
        else:
            raise ValueError("Unidentified set of sensors")

        row, col = np.diag_indices(D.shape[0])

        D[row[:3], col[:3]] = cs_dict['env_trq_sigma']**2
        D[row[3:6], col[3:6]] = cs_dict['ekf_Q_sigma_m_res']**2
        D[row[6:9], col[6:9]] = cs_dict['ekf_Q_sigma_b_bias']**2
        if self.sensors_used in (FilterType.MTM_GYRO, FilterType.MTM_GYRO_SUN):
            D[row[9:12], col[9:12]] = cs_dict['ekf_Q_sigma_g_bias']**2

        return G.dot(D).dot(G.T) * self.tau

    def ini_measurements_covariance(self, cs_dict):

        if self.sensors_used == FilterType.MTM:
            R_cov = np.eye(3) * cs_dict['magnetometer_noise']**2
        elif self.sensors_used == FilterType.MTM_GYRO:
            R_cov = np.eye(6)
            R_cov[0, 0] = R_cov[1, 1] = R_cov[2, 2] = cs_dict['magnetometer_noise']**2
            R_cov[3, 3] = R_cov[4, 4] = R_cov[5, 5] = cs_dict['gyro_noise']**2
        elif self.sensors_used == FilterType.MTM_GYRO_SUN:
            R_cov = np.eye(9)
            R_cov[0, 0] = R_cov[1, 1] = R_cov[2, 2] = cs_dict['magnetometer_noise']**2
            R_cov[3, 3] = R_cov[4, 4] = R_cov[5, 5] = cs_dict['gyro_noise']**2
            R_cov[6, 6] = R_cov[7, 7] = R_cov[8, 8] = cs_dict['sun_sensor_noise']**2
        else:
            raise ValueError("Unidentified set of sensors")

        return R_cov

    def evolution_matrix(self, state, b_model_body0):
        q = state[:4]
        omega = state[4:7]

        e3 = math_utils.rotate_vec_with_quat(q, np.array([0., 0., 1.]))

        F_gr = 6 * (self.sc.orb.mean_motion ** 2) * (
                math_utils.skew_symm(e3).dot(self.sc.J).dot(math_utils.skew_symm(e3)) -
                math_utils.skew_symm(self.sc.J.dot(e3)).dot(math_utils.skew_symm(e3)))

        F_gyr = math_utils.skew_symm(self.sc.J.dot(omega)) - math_utils.skew_symm(omega).dot(self.sc.J)

        F_ctrl = 2 * math_utils.skew_symm(self.sc.ctrl.m + self.sc.m_res).dot(math_utils.skew_symm(b_model_body0))

        F1 = np.concatenate((-math_utils.skew_symm(omega), 0.5 * np.eye(3)), 1)
        F2 = np.concatenate((self.sc.J_inv.dot(F_gr + F_ctrl), self.sc.J_inv.dot(F_gyr)), 1)

        sz = self.state_sz - 1

        F = np.zeros((sz, sz))
        F[:6, :6] = np.concatenate((F1, F2), 0)
        F[3:6, 6:9] = -self.sc.J_inv.dot(math_utils.skew_symm(b_model_body0))

        return np.eye(sz) + F * self.tau

    def observation_matrix(self, e_b_model, e_sun_model=None):
        if self.sensors_used == FilterType.MTM:
            return np.concatenate((2 * math_utils.skew_symm(e_b_model), np.zeros((3, 6)), np.eye(3)), 1)
        elif self.sensors_used == FilterType.MTM_GYRO:
            H_magn = np.concatenate((2 * math_utils.skew_symm(e_b_model),
                                     np.zeros((3, 6)), np.eye(3), np.zeros((3, 3))), 1)
            H_gyro = np.concatenate((np.zeros((3, 3)), np.eye(3), np.zeros((3, 6)), np.eye(3)), 1)
            return np.concatenate((H_magn, H_gyro), 0)
        elif self.sensors_used == FilterType.MTM_GYRO_SUN:
            H_magn = np.concatenate((2 * math_utils.skew_symm(e_b_model),
                                     np.zeros((3, 6)), np.eye(3), np.zeros((3, 3))), 1)
            H_gyro = np.concatenate((np.zeros((3, 3)), np.eye(3), np.zeros((3, 6)), np.eye(3)), 1)
            H_sun = np.concatenate((2 * math_utils.skew_symm(e_sun_model), np.zeros((3, 12))), 1)
            return np.concatenate((H_magn, H_gyro, H_sun), 0)
        else:
            raise ValueError("Unidentified set of sensors")

    def kalman_gain(self, P, H, b_model_norm, e_sun_norm=1):
        if self.sensors_used == FilterType.MTM:
            corr_matr = np.eye(3) / (b_model_norm ** 2)
        elif self.sensors_used == FilterType.MTM_GYRO:
            corr_matr = np.eye(6)
            corr_matr[:3, :3] = np.eye(3) / (b_model_norm ** 2)
        elif self.sensors_used == FilterType.MTM_GYRO_SUN:
            epsilon = 1e-5
            corr_matr = np.eye(9)
            corr_matr[:3, :3] = np.eye(3) / (b_model_norm ** 2)
            corr_matr[6:, 6:] = np.eye(3) / ((e_sun_norm + epsilon)** 2)
        else:
            raise ValueError("Unidentified set of sensors")

        R_corr = self.R_cov.dot(corr_matr)
        S = np.linalg.inv(H.dot(P).dot(H.T) + R_corr)

        return P.dot(H.T).dot(S)

    def kalman_propage(self, time, integration_time, state, m_ctrl):

        return self.sc.propage(time, integration_time, state, m_ctrl, onboard_model=True)

    def prediction(self, time, x_prev, b_model_body_0):

        x_pred = np.zeros(self.state_sz)
        state = x_prev[:7]
        state_ctrl = self.kalman_propage(time, self.sc.ctrl.T_ctrl, state, self.sc.ctrl.m)
        time += self.sc.ctrl.T_ctrl
        x_pred[0:7] = self.kalman_propage(time, self.sc.ctrl.T_meas, state_ctrl, np.array([0., 0., 0.]))
        x_pred[7:] = x_prev[7:]

        Phi = self.evolution_matrix(x_prev, b_model_body_0)
        P_pred = Phi.dot(self.P_est).dot(Phi.T) + self.Q0

        return x_pred, P_pred

    def correction(self, x_pred, P_pred, b_model_T, b_sensor, omega_sensor=None, e_sun_model=None, e_sun_sensor=None):
        def vec2unit_quat(vec):
            if np.linalg.norm(vec) < 1.:
                return np.array([np.sqrt(1. - np.linalg.norm(vec)**2), vec[0], vec[1], vec[2]])
            else:
                return math_utils.normalize(np.array([0., vec[0], vec[1], vec[2]]))

        b_model_body = math_utils.rotate_vec_with_quat(x_pred[:4], b_model_T)
        b_model_norm = np.linalg.norm(b_model_T)
        magn_bias_est = x_pred[10:13]
        e_sun_body = None
        e_sun_norm = 0.

        if self.sensors_used == FilterType.MTM:
            z = b_sensor / b_model_norm
            Hx = (b_model_body + magn_bias_est) / b_model_norm
        elif self.sensors_used == FilterType.MTM_GYRO:
            z = np.concatenate((b_sensor / b_model_norm, omega_sensor), 0)
            gyro_bias_est = x_pred[13:16]
            Hx = np.concatenate(((b_model_body + magn_bias_est) / b_model_norm, x_pred[4:7] + gyro_bias_est), 0)
        elif self.sensors_used == FilterType.MTM_GYRO_SUN:
            e_sun_body = math_utils.rotate_vec_with_quat(x_pred[:4], e_sun_model)
            e_sun_norm = np.linalg.norm(e_sun_body)
            z = np.concatenate((b_sensor / b_model_norm, omega_sensor, e_sun_sensor), 0)
            gyro_bias_est = x_pred[13:16]
            Hx = np.concatenate(((b_model_body + magn_bias_est) / b_model_norm,
                                 x_pred[4:7] + gyro_bias_est, e_sun_body), 0)
        else:
            raise ValueError("Unidentified set of sensors")

        H = self.observation_matrix(b_model_body / b_model_norm, e_sun_body)
        K = self.kalman_gain(P_pred, H, b_model_norm, e_sun_norm)

        x_cor = K.dot(z - Hx)
        q_cor = vec2unit_quat(x_cor[:3])
        x_cor[9:12] *= b_model_norm

        x_est = np.zeros(self.state_sz)
        x_est[:4] = math_utils.normalize(math_utils.quat_product(x_pred[:4], q_cor))
        x_est[4:] = x_pred[4:] + x_cor[3:]

        self.P_est = (np.eye(self.state_sz - 1) - K.dot(H)).dot(P_pred)

        return x_est

    def kalman_estimate(self, time, x_prev, m_ctrl, b_model, b_sensor, omega_sensor=None, sun_model=None,
                        sun_sensor=None):
        self.sc.ctrl.m = m_ctrl
        self.sc.m_res = x_prev[7:10]
        b_model_body_0 = math_utils.rotate_vec_with_quat(x_prev[:4], b_model[:, 0])

        x_pred, P_pred = self.prediction(time, x_prev, b_model_body_0)
        x_est = self.correction(x_pred, P_pred, b_model[:, 1], b_sensor, omega_sensor, sun_model, sun_sensor)

        return x_est


class Spacecraft:
    def __init__(self, cs_dict):
        self.J = cs_dict['inertia_tensor']
        self.J_inv = np.linalg.inv(self.J)
        self.start_time = cs_dict['start_time']

        if cs_dict['use_magnetometer']:
            self.mtm = Magnetometer(cs_dict['magnetometer_bias'],
                                    cs_dict['magnetometer_noise'],
                                    cs_dict['magnetometer_pos'])

        if cs_dict['use_gyro']:
            self.gyro = Gyroscope(cs_dict['gyro_bias'], cs_dict['gyro_noise'])

        if cs_dict['use_sun_sensors']:
            self.sun_sensors = []
            ss_list = cs_dict['sun_sensors_axes']

            for i in range(len(ss_list[:, 0])):
                self. sun_sensors.append(SunSensor(ss_list[i, :],
                                                   cs_dict['sun_sensor_fov'],
                                                   cs_dict['sun_sensor_noise']))

        V_max       = cs_dict['mtq_V_max']
        resistance  = cs_dict['mtq_coil_resistance']
        alpha       = cs_dict['mtq_alpha']
        n_turns     = cs_dict['mtq_n_turns']
        area        = cs_dict['mtq_coil_area']
        noise_sigma = cs_dict['mtq_noise_sigma']
        self.m_max = (V_max / resistance) * area * n_turns
        self.m2V = V_max / self.m_max
        self.mtq_x = Magnetorquer(np.array([1., 0., 0.]), V_max, resistance, alpha, area, n_turns, noise_sigma)
        self.mtq_y = Magnetorquer(np.array([0., 1., 0.]), V_max, resistance, alpha, area, n_turns, noise_sigma)
        self.mtq_z = Magnetorquer(np.array([0., 0., 1.]), V_max, resistance, alpha, area, n_turns, noise_sigma)

        self.orb = CircularOrbit(cs_dict['altitude'], cs_dict['inclination'])

        self.ctrl = Parameters()
        self.ctrl.Kw = cs_dict['pd_Kw_coef'] / self.orb.mean_motion  # [N * m * s / T^2] D-gain for PD-controller
        self.ctrl.Ka = cs_dict['pd_Ka']  # [N * m / T^2] P-gain for PD-controller
        self.ctrl.T_meas = cs_dict['ctrl_meas_time']  # [s] sampling time step
        self.ctrl.T_ctrl = cs_dict['ctrl_ctrl_time']  # [s] control time step
        self.ctrl.T_loop = self.ctrl.T_meas + self.ctrl.T_ctrl
        self.ctrl.m = np.array([0., 0., 0.])

        # residual magnetic dipole
        self.m_res = np.array([0., 0., 0.])
        self.m_res_const = cs_dict['const_m_res'] * np.array([1, 1, 1]) / np.sqrt(3)  # permanent part of rmm
        self.m_res_pos = np.array([0., 0., 0.])  # [m] the place of residual magnetic moment

        self.env = Environment(self.orb.R,
                               cs_dict['env_trq_sigma'],
                               cs_dict['env_magn_field_sigma'], cs_dict['start_time'], self.orb.eci2orb)

        _, self.e_sun_eci = self.env.sun_direction_eci(self.start_time)

    def is_eclipsed(self, time):
        e_sc_pos, _ = self.orb.unit_position_velocity(self.orb.mean_motion * time)
        r_sc_pos = e_sc_pos * self.orb.R
        r_sun, _ = self.env.sun_direction_eci(self.start_time)

        return not self.env.sight(r_sun, r_sc_pos)

    def read_sun_sensors(self, time, q_ob):
        e_sun_orb = self.orb.eci2orb(self.orb.mean_motion * time).dot(self.e_sun_eci)
        e_sun_sens = np.array([0., 0., 0.])

        if self.is_eclipsed(time):
            return 0. * e_sun_orb, e_sun_sens
        else:
            max_intensity = 0.
            e_sun_b = math_utils.rotate_vec_with_quat(q_ob, e_sun_orb)
            for i in range(len(self.sun_sensors)):
                e_sun, intensity = self.sun_sensors[i].ss_readings(e_sun_b)
                if intensity > max_intensity:
                    max_intensity = intensity
                    e_sun_sens = e_sun
            return e_sun_orb, e_sun_sens

    def update_ctrl_moment(self, quat, omega_rel, magnetic_field, m_res_est):

        if quat[0] == 0.:
            multiplier = 1.
        else:
            multiplier = np.sign(quat[0])

        S = 4. * quat[1:] * multiplier

        self.ctrl.m = -self.ctrl.Kw * math_utils.cross_product(magnetic_field, omega_rel) \
                      - self.ctrl.Ka * math_utils.cross_product(magnetic_field, S) - m_res_est

        self.ctrl.m *= (self.ctrl.T_loop / self.ctrl.T_ctrl)

        if max(abs(self.ctrl.m)) > self.m_max:
            self.ctrl.m *= self.m_max / max(abs(self.ctrl.m))

        return

    def actuate_ctrl_moment(self):

        v_ctrl_req = self.m2V * self.ctrl.m

        m_ctrl = self.mtq_x.magnetic_moment(v_ctrl_req[0]) + \
                 self.mtq_y.magnetic_moment(v_ctrl_req[1]) + \
                 self.mtq_z.magnetic_moment(v_ctrl_req[2])

        return m_ctrl

    def rmm(self):
        m_local = self.m_res_const
        r_local = self.m_res_pos - self.mtm.pos
        e_r_local = r_local / np.linalg.norm(r_local)

        b_bias_local = (self.env.B_1 / (np.linalg.norm(r_local) ** 3.)) * \
                       (3. * m_local.dot(e_r_local) * e_r_local - m_local)

        self.m_res = m_local

        return m_local, b_bias_local

    def propage(self, t, T, x_0, m_ctrl, onboard_model=False):

        def rhs(tau, x):

            q = math_utils.normalize(x[:4])
            omega = x[4:7]

            omega_rel = omega - math_utils.rotate_vec_with_quat(q, np.array([0., self.orb.mean_motion, 0.]))
            B_IGRF = self.env.direct_dipole_(self.orb.mean_motion * (t + tau),
                                                        self.orb.incl, (t + tau))
            b_orb = self.env.direct_dipole(self.orb.mean_motion * (t + tau), self.orb.incl)
            b_orb = B_IGRF
            b_body = math_utils.rotate_vec_with_quat(q, b_orb)

            if onboard_model:
                b_env = b_body
            else:
                b_env = self.env.external_magnetic_field(b_body)

            m = m_ctrl + self.m_res
            trq = self.env.external_torque(q, self.orb.mean_motion, self.J, (not onboard_model)) + \
                  self.env.magnetic_torque(m, b_env)

            x_dot = np.zeros(7)

            x_dot[4:] = self.J_inv.dot(trq - math_utils.cross_product(omega, self.J.dot(omega)))
            x_dot[0] = -0.5 * q[1:].dot(omega_rel)
            x_dot[1:4] = 0.5 * (q[0] * omega_rel + math_utils.cross_product(q[1:], omega_rel))

            return x_dot

        x_T = solve_ivp(rhs, (0, T), x_0).y[:, -1]
        return x_T

    def triad(self, time, q_ob):
        def get_matrix_from2vecs(vec1, vec2):
            M = np.zeros((3, 3))
            e_x = vec1
            e_y = np.cross(vec1, vec2)
            e_z = np.cross(e_x, e_y)
            M[:, 0] = math_utils.normalize(e_x)
            M[:, 1] = math_utils.normalize(e_y)
            M[:, 2] = math_utils.normalize(e_z)
            return M

        if self.is_eclipsed(time):
            return None

        e_sun_orb, e_sun_sensor = self.read_sun_sensors(time, q_ob)

        if np.linalg.norm(e_sun_sensor) == 0:
            return None

        m_res, mtm_bias_rm = self.rmm()
        B_IGRF = self.env.direct_dipole(self.orb.mean_motion * time, self.orb.incl, time)
        b_orb = self.env.direct_dipole(self.orb.mean_motion * time, self.orb.incl)
        b_orb = B_IGRF
        b_env = self.env.external_magnetic_field(math_utils.rotate_vec_with_quat(q_ob, b_orb))
        b_sens = self.mtm.magnetometer_readings(b_env + mtm_bias_rm)
        e_magn_orb = math_utils.normalize(b_orb)
        e_magn_sensor = math_utils.normalize(b_sens)

        matrix_b = get_matrix_from2vecs(e_sun_sensor, e_magn_sensor)
        matrix_o = get_matrix_from2vecs(e_sun_orb, e_magn_orb)

        att_matrix = matrix_o.dot(matrix_b.T)
        q = math_utils.matr2quat(att_matrix)
        return q


class ControlSystem:
    def __init__(self, param_dict):
        self.spacecraft = Spacecraft(param_dict)
        self.filter = param_dict['do_filter']
        if self.filter:
            self.EKF = ExtendedKalmanFilter(param_dict)
            self.state_est_sz = self.EKF.state_sz
        else:
            self.state_est_sz = 7  # quaternion (4) and angular velocity (3)

    def control_loop(self, t0, x0, x_est0):
        # control
        time = t0
        state = x0

        m_res, mtm_bias_rm = self.spacecraft.rmm()
        m_ctrl = self.spacecraft.actuate_ctrl_moment()
        state_ctrl = self.spacecraft.propage(time, self.spacecraft.ctrl.T_ctrl, state, m_ctrl)
        time += self.spacecraft.ctrl.T_ctrl

        # measurements
        stateT = self.spacecraft.propage(time, self.spacecraft.ctrl.T_meas, state_ctrl, np.array([0., 0., 0.]))
        time += self.spacecraft.ctrl.T_meas
        q = math_utils.normalize(stateT[:4])
        stateT[:4] = q
        B_IGRF = self.spacecraft.env.direct_dipole_(self.spacecraft.orb.mean_motion * time, self.spacecraft.orb.incl, time)
        b_orb = self.spacecraft.env.direct_dipole(self.spacecraft.orb.mean_motion * time, self.spacecraft.orb.incl)
        b_orb = B_IGRF
        b_env = self.spacecraft.env.external_magnetic_field(math_utils.rotate_vec_with_quat(q, b_orb))
        b_sens = self.spacecraft.mtm.magnetometer_readings(b_env + mtm_bias_rm)

        # EKF
        if self.filter:
            g_sens = None
            e_sun_sens = None
            e_sun_orb = None

            if self.EKF.sensors_used in (FilterType.MTM_GYRO, FilterType.MTM_GYRO_SUN):
                g_sens = self.spacecraft.gyro.gyro_readings(stateT[4:7])

                if self.EKF.sensors_used == FilterType.MTM_GYRO_SUN:
                    e_sun_orb, e_sun_sens = self.spacecraft.read_sun_sensors(time, q)

            b_model = np.zeros((3, 2))
            b_model[:, 0] = self.spacecraft.env.direct_dipole(self.spacecraft.orb.mean_motion * t0,
                                                              self.spacecraft.orb.incl)
            b_model[:, 1] = b_orb
            m_ctrl = self.spacecraft.ctrl.m

            x_estT = self.EKF.kalman_estimate(t0, x_est0, m_ctrl, b_model, b_sens, g_sens, e_sun_orb, e_sun_sens)
        else:
            x_estT = np.zeros(self.state_est_sz)
            x_estT[:7] = stateT[:7]

        # control moment
        q_est = x_estT[:4]
        omega_est = x_estT[4:7]

        omega_rel_est = omega_est - math_utils.rotate_vec_with_quat(q_est,
                                                                    np.array([0., self.spacecraft.orb.mean_motion, 0.]))
        if self.filter:
            m_res_est, m_bias_est = x_estT[7:10], x_estT[10:13]
        else:
            m_res_est, m_bias_est = m_res, mtm_bias_rm

        self.spacecraft.update_ctrl_moment(q_est, omega_rel_est, b_sens - m_bias_est, m_res_est)

        telemetry = []

        return time, stateT, x_estT, omega_rel_est, m_ctrl, telemetry
