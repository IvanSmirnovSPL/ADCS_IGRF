import numpy as np
import sys


from matplotlib import pyplot as plt

from scipy.special import binom
from scipy.special import factorial as fact

# matplotlib inline

IGRF_COEFS_TABLE = np.genfromtxt('./IGRF/igrf13coeffs.txt', dtype=str)[1:]

X_CIP_COEFS_TABLES = [np.genfromtxt('./IGRF/tab5.2a.txt', dtype=float, skip_header=36, skip_footer=298),
                      np.genfromtxt('./IGRF/tab5.2a.txt', dtype=float, skip_header=1345, skip_footer=44),
                      np.genfromtxt('./IGRF/tab5.2a.txt', dtype=float, skip_header=1601, skip_footer=7),
                      np.genfromtxt('./IGRF/tab5.2a.txt', dtype=float, skip_header=1640, skip_footer=2),
                      np.genfromtxt('./IGRF/tab5.2a.txt', dtype=float, skip_header=1647)]

Y_CIP_COEFS_TABLES = [np.genfromtxt('./IGRF/tab5.2b.txt', dtype=float, skip_header=35, skip_footer=317),
                      np.genfromtxt('./IGRF/tab5.2b.txt', dtype=float, skip_header=1000, skip_footer=39),
                      np.genfromtxt('./IGRF/tab5.2b.txt', dtype=float, skip_header=1280, skip_footer=8),
                      np.genfromtxt('./IGRF/tab5.2b.txt', dtype=float, skip_header=1313, skip_footer=2),
                      np.genfromtxt('./IGRF/tab5.2b.txt', dtype=float, skip_header=1321)]

sXY_CIO_COEFS_TABLES = [np.genfromtxt('./IGRF/tab5.2c.txt', dtype=float, skip_header=40, skip_footer=37),
                        np.genfromtxt('./IGRF/tab5.2c.txt', dtype=float, skip_header=76, skip_footer=33),
                        np.genfromtxt('./IGRF/tab5.2c.txt', dtype=float, skip_header=82, skip_footer=7),
                        np.genfromtxt('./IGRF/tab5.2c.txt', dtype=float, skip_header=110, skip_footer=2),
                        np.genfromtxt('./IGRF/tab5.2c.txt', dtype=float, skip_header=117)]

# TIP_AND_DUT1_TABLE = np.genfromtxt('./6_BULLETIN_A_V2013_016.txt', dtype=float, skip_header=122, skip_footer=42)[:, 3:]
TIP_AND_DUT1_TABLE = np.genfromtxt('./IGRF/6_BULLETIN_A.txt', dtype=float, skip_header=141, skip_footer=112)[:, 3:]

def assoc_legendre(theta, n, m):
    sum = 0

    for k in range(m, n + 1):
        sum += fact(k) / fact(k - m) * np.cos(theta) ** (k - m) * binom(n, k) * binom((n + k - 1) / 2, n)

    return (-1) ** m * 2 ** n * np.sin(theta) ** m * sum


def theta_deriv(theta, n, m):
    SUM = 0
    sum = 0

    if m != 0:

        for k in range(m, n + 1):
            sum += fact(k) / fact(k - m) * np.cos(theta) ** (k - m) * binom(n, k) * binom((n + k - 1) / 2, n)

        SUM += m * np.sin(theta) ** (m - 1) * np.cos(theta) * sum

        sum = 0

    for k in range(m + 1, n + 1):
        sum += fact(k) / fact(k - m - 1) * np.cos(theta) ** (k - m - 1) * (-np.sin(theta)) * binom(n, k) * binom(
            (n + k - 1) / 2, n)

    SUM += np.sin(theta) ** m * sum

    return (-1) ** m * 2 ** n * SUM


def assoc_legendre_norm(theta, n, m):
    if m == 0:

        return assoc_legendre(theta, n, m)

    else:

        return (-1) ** m * np.sqrt(2 * fact(n - m) / fact(n + m)) * assoc_legendre(theta, n, m)


def theta_deriv_norm(theta, n, m):
    if m == 0:

        return theta_deriv(theta, n, m)

    else:

        return (-1) ** m * np.sqrt(2 * fact(n - m) / fact(n + m)) * theta_deriv(theta, n, m)


def theta_deriv_full(a, G, H, r, theta, phi):
    N = 13

    sum = 0

    for n in range(1, N + 1):

        for m in range(n + 1):
            sum += (a / r) ** (n + 1) * (G[n, m] * np.cos(m * phi) + H[n, m] * np.sin(m * phi)) * theta_deriv_norm(
                theta, n, m)

    return sum * a


def r_deriv_full(a, G, H, r, theta, phi):
    N = 13

    sum = 0

    for n in range(1, N + 1):

        for m in range(n + 1):
            sum += (n + 1) * (a / r) ** (n + 2) * (G[n, m] * np.cos(m * phi) + H[n, m] * np.sin(m * phi)) * \
                   assoc_legendre_norm(theta, n, m)

    return sum * (-1)


def phi_deriv_full(a, G, H, r, theta, phi):
    N = 13

    sum = 0

    for n in range(1, N + 1):

        for m in range(n + 1):
            sum += (a / r) ** (n + 1) * m * (
                        H[n, m] * np.cos(m * phi) - G[n, m] * np.sin(m * phi)) * assoc_legendre_norm(theta, n, m)

    return sum * a


def IGRF_coef(year, table=IGRF_COEFS_TABLE):
    if (year < 1900.0) or (year >= 2025.0):
        raise Exception('Invalid year, should be in [1900, 2025)')

    delta_year = year % 5
    year_5 = year - delta_year

    G = np.zeros((14, 14))
    H = np.zeros((14, 14))

    if year_5 < 2020:

        col_num = int(year_5 / 5 - 377)

        coefs = table[:, (0, 1, 2, col_num, col_num + 1)]

        for line in coefs[1:]:

            val = float(line[3]) + delta_year / 5 * (float(line[4]) - float(line[3]))

            if line[0] == 'g':

                G[int(line[1]), int(line[2])] = val

            elif line[0] == 'h':

                H[int(line[1]), int(line[2])] = val

            else:

                raise Exception('Incorrect symbol in line, should be \'g/h\'')

    else:

        coefs = table[:, (0, 1, 2, -2, -1)]

        for line in coefs[1:]:

            val = float(line[3]) + delta_year * float(line[4])

            if line[0] == 'g':

                G[int(line[1]), int(line[2])] = val

            elif line[0] == 'h':

                H[int(line[1]), int(line[2])] = val

            else:

                raise Exception('Incorrect symbol in line, should be \'g/h\'')

    return G, H


def magn_field(year, r, theta, phi, table=IGRF_COEFS_TABLE):
    a = 6371200  # [m]

    G, H = IGRF_coef(year, table=table)

    B_r = - r_deriv_full(a, G, H, r, theta, phi)
    B_theta = - theta_deriv_full(a, G, H, r, theta, phi) / r
    B_phi = - phi_deriv_full(a, G, H, r, theta, phi) / (r * np.sin(theta))

    return np.array([B_theta, B_phi, B_r])




def DCM_SEU_to_ECEF(theta, phi):
    return np.array([[np.cos(theta) * np.cos(phi), -np.sin(phi), np.sin(theta) * np.cos(phi)],
                     [np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)],
                     [-np.sin(theta), 0, np.cos(theta)]])


def magn_field_ECEF(year, r, theta, phi, table=IGRF_COEFS_TABLE):
    B = magn_field(year, r, theta, phi, table=table)

    A = DCM_SEU_to_ECEF(theta, phi)

    return A @ B


# %%


def F1(JC):
    """
    l (Mean Anomaly of the Moon) in [rad]
    """

    DEG2RAD = np.pi / 180
    ARCSEC2RAD = DEG2RAD / 3600

    return 134.96340251 * DEG2RAD + \
           (1717915923.2178 * JC + 31.8792 * JC ** 2 + 0.051635 * JC ** 3 - 0.00024470 * JC ** 4) * ARCSEC2RAD


def F2(JC):
    """
    l' (Mean Anomaly of the Sun) in [rad]
    """

    DEG2RAD = np.pi / 180
    ARCSEC2RAD = DEG2RAD / 3600

    return 357.52910918 * DEG2RAD + \
           (129596581.0481 * JC - 0.5532 * JC ** 2 + 0.000136 * JC ** 3 - 0.00001149 * JC ** 4) * ARCSEC2RAD


def F3(JC):
    """
    F = L - Omega (Mean Longitude of the Moon - Mean Longitude of the Ascending Node of the Moon) in [rad]
    """

    DEG2RAD = np.pi / 180
    ARCSEC2RAD = DEG2RAD / 3600

    return 93.27209062 * DEG2RAD + \
           (1739527262.8478 * JC - 12.7512 * JC ** 2 - 0.001037 * JC ** 3 + 0.00000417 * JC ** 4) * ARCSEC2RAD


def F4(JC):
    """
    D (Mean Elongation of the Moon from the Sun) in [rad]
    """

    DEG2RAD = np.pi / 180
    ARCSEC2RAD = DEG2RAD / 3600

    return 297.85019547 * DEG2RAD + \
           (1602961601.2090 * JC - 6.3706 * JC ** 2 + 0.006593 * JC ** 3 - 0.00003169 * JC ** 4) * ARCSEC2RAD


def F5(JC):
    """
    Omega (Mean Longitude of the Ascending Node of the Moon) in [rad]
    """

    DEG2RAD = np.pi / 180
    ARCSEC2RAD = DEG2RAD / 3600

    return 125.04455501 * DEG2RAD + \
           (-6962890.5431 * JC + 7.4722 * JC ** 2 + 0.007702 * JC ** 3 - 0.00005939 * JC ** 4) * ARCSEC2RAD


def F6(JC):
    """
    L_me (Mean Longitude of the Mercury) in [rad]
    """

    return 4.402608842 + 2608.7903141574 * JC


def F7(JC):
    """
    L_ve (Mean Longitude of the Venus) in [rad]
    """

    return 3.176146697 + 1021.3285546211 * JC


def F8(JC):
    """
    L_e (Mean Longitude of the Earth) in [rad]
    """

    return 1.753470314 + 628.3075849991 * JC


def F9(JC):
    """
    L_ma (Mean Longitude of the Mars) in [rad]
    """

    return 6.203480913 + 334.0612426700 * JC


def F10(JC):
    """
    L_ju (Mean Longitude of the Jupiter) in [rad]
    """

    return 0.599546497 + 52.9690962641 * JC


def F11(JC):
    """
    L_sa (Mean Longitude of the Saturn) in [rad]
    """

    return 0.874016757 + 21.3299104960 * JC


def F12(JC):
    """
    L_ur (Mean Longitude of the Uranus) in [rad]
    """

    return 5.481293872 + 7.4781598567 * JC


def F13(JC):
    """
    L_ne (Mean Longitude of the Neptune) in [rad]
    """

    return 5.311886287 + 3.8133035638 * JC


def F14(JC):
    """
    pA (General precession) in [rad]
    """

    return 0.02438175 * JC + 0.00000538691 * JC ** 2




def nutation_argument(N, JC):
    F = np.array([F1(JC), F2(JC), F3(JC), F4(JC), F5(JC),
                  F6(JC), F7(JC), F8(JC), F9(JC), F10(JC), F11(JC), F12(JC), F13(JC), F14(JC)])

    return N @ F



def X_polynom(JC):
    """
    Polynomial part of the X-coordinate of the CIP in [uas]
    """

    return -16616.99 + 2004191742.88 * JC - 427219.05 * JC ** 2 - 198620.54 * JC ** 3 - 46.05 * JC ** 4 + 5.98 * JC ** 5


def X_non_polynom(JC, tables=X_CIP_COEFS_TABLES):
    """
    Non-polynomial part of the X-coordinate of the CIP in [uas]
    """

    coefs_0, coefs_1, coefs_2, coefs_3, coefs_4 = tables

    a_s0 = coefs_0[:, 1]
    a_s1 = coefs_1[:, 1]
    a_s2 = coefs_2[:, 1]
    a_s3 = coefs_3[:, 1]
    a_s4 = coefs_4[1].reshape(1, )
    a_sj = [a_s0, a_s1, a_s2, a_s3, a_s4]

    a_c0 = coefs_0[:, 2]
    a_c1 = coefs_1[:, 2]
    a_c2 = coefs_2[:, 2]
    a_c3 = coefs_3[:, 2]
    a_c4 = coefs_4[2].reshape(1, )
    a_cj = [a_c0, a_c1, a_c2, a_c3, a_c4]

    N0 = coefs_0[:, 3:]
    N1 = coefs_1[:, 3:]
    N2 = coefs_2[:, 3:]
    N3 = coefs_3[:, 3:]
    N4 = coefs_4[3:].reshape(1, 14)
    Nj = [N0, N1, N2, N3, N4]

    SUM = 0

    for (j, (a_s, a_c, N)) in enumerate(zip(a_sj, a_cj, Nj)):

        beta = nutation_argument(N, JC)

        for (a_si, a_ci, betai) in zip(a_s, a_c, beta):
            SUM += (a_si * np.sin(betai) + a_ci * np.cos(betai)) * JC ** j

    return SUM


def X_CIP(JC, tables=X_CIP_COEFS_TABLES):
    """
    X-coordinate of the CIP in [rad]
    """

    UARCSEC2RAD = np.pi / 180 / 3600 / 1000000

    return (X_polynom(JC) + X_non_polynom(JC, tables=tables)) * UARCSEC2RAD



def Y_polynom(JC):
    """
    Polynomial part of the Y-coordinate of the CIP in [uas]
    """

    return -6950.78 - 25381.99 * JC - 22407250.99 * JC ** 2 + 1842.28 * JC ** 3 + 1113.06 * JC ** 4 + 0.99 * JC ** 5


def Y_non_polynom(JC, tables=Y_CIP_COEFS_TABLES):
    """
    Non-polynomial part of the Y-coordinate of the CIP in [uas]
    """

    coefs_0, coefs_1, coefs_2, coefs_3, coefs_4 = tables

    b_s0 = coefs_0[:, 1]
    b_s1 = coefs_1[:, 1]
    b_s2 = coefs_2[:, 1]
    b_s3 = coefs_3[:, 1]
    b_s4 = coefs_4[1].reshape(1, )
    b_sj = [b_s0, b_s1, b_s2, b_s3, b_s4]

    b_c0 = coefs_0[:, 2]
    b_c1 = coefs_1[:, 2]
    b_c2 = coefs_2[:, 2]
    b_c3 = coefs_3[:, 2]
    b_c4 = coefs_4[2].reshape(1, )
    b_cj = [b_c0, b_c1, b_c2, b_c3, b_c4]

    N0 = coefs_0[:, 3:]
    N1 = coefs_1[:, 3:]
    N2 = coefs_2[:, 3:]
    N3 = coefs_3[:, 3:]
    N4 = coefs_4[3:].reshape(1, 14)
    Nj = [N0, N1, N2, N3, N4]

    SUM = 0

    for (j, (b_s, b_c, N)) in enumerate(zip(b_sj, b_cj, Nj)):

        beta = nutation_argument(N, JC)

        for (b_si, b_ci, betai) in zip(b_s, b_c, beta):
            SUM += (b_si * np.sin(betai) + b_ci * np.cos(betai)) * JC ** j

    return SUM


def Y_CIP(JC, tables=Y_CIP_COEFS_TABLES):
    """
    Y-coordinate of the CIP in [rad]
    """

    UARCSEC2RAD = np.pi / 180 / 3600 / 1000000

    return (Y_polynom(JC) + Y_non_polynom(JC, tables=tables)) * UARCSEC2RAD


def sXY_polynom(JC):
    """
    Polynomial part of the s + XY/2 quantity in [uas]
    """

    return -6950.78 - 25381.99 * JC - 22407250.99 * JC ** 2 + 1842.28 * JC ** 3 + 1113.06 * JC ** 4 + 0.99 * JC ** 5


def sXY_non_polynom(JC, tables=sXY_CIO_COEFS_TABLES):
    """
    Non-polynomial part of the s + XY/2 quantity in [uas]
    """

    coefs_0, coefs_1, coefs_2, coefs_3, coefs_4 = tables

    c_s0 = coefs_0[:, 1]
    c_s1 = coefs_1[:, 1]
    c_s2 = coefs_2[:, 1]
    c_s3 = coefs_3[:, 1]
    c_s4 = coefs_4[1].reshape(1, )
    c_sj = [c_s0, c_s1, c_s2, c_s3, c_s4]

    c_c0 = coefs_0[:, 2]
    c_c1 = coefs_1[:, 2]
    c_c2 = coefs_2[:, 2]
    c_c3 = coefs_3[:, 2]
    c_c4 = coefs_4[2].reshape(1, )
    c_cj = [c_c0, c_c1, c_c2, c_c3, c_c4]

    N0 = coefs_0[:, 3:]
    N1 = coefs_1[:, 3:]
    N2 = coefs_2[:, 3:]
    N3 = coefs_3[:, 3:]
    N4 = coefs_4[3:].reshape(1, 14)
    Nj = [N0, N1, N2, N3, N4]

    SUM = 0

    for (j, (c_s, c_c, N)) in enumerate(zip(c_sj, c_cj, Nj)):

        beta = nutation_argument(N, JC)

        for (c_si, c_ci, betai) in zip(c_s, c_c, beta):
            SUM += (c_si * np.sin(betai) + c_ci * np.cos(betai)) * JC ** j

    return SUM


def sXY(JC, tables=sXY_CIO_COEFS_TABLES):
    """
    s + XY/2 quantity in [rad]
    """

    UARCSEC2RAD = np.pi / 180 / 3600 / 1000000

    return (sXY_polynom(JC) + sXY_non_polynom(JC, tables=tables)) * UARCSEC2RAD


def nut_prec(JC, tables_X=X_CIP_COEFS_TABLES, tables_Y=Y_CIP_COEFS_TABLES, tables_sXY=sXY_CIO_COEFS_TABLES):
    X = X_CIP(JC, tables=tables_X)
    Y = Y_CIP(JC, tables=tables_Y)
    s = sXY(JC, tables=tables_sXY) - X * Y / 2

    X2Y2 = X ** 2 + Y ** 2
    XY = X * Y

    a = 0.5 + X2Y2 / 8

    A = np.array([[1 - a * X ** 2, -a * XY, X],
                  [-a * XY, 1 - a * Y ** 2, Y],
                  [-X, -Y, 1 - a * X2Y2]])

    R = np.array([[np.cos(s), np.sin(s), 0],
                  [-np.sin(s), np.cos(s), 0],
                  [0, 0, 1]])

    return A @ R

def greg_to_julian(GD):
    year, month, day, hour, min, sec = GD

    if month <= 2:
        year -= 1
        month += 12

    return int(365.25 * year) + int(
        30.6001 * (month + 1)) + day + 1720981.5 + hour / 24 + min / 24 / 60 + sec / 24 / 3600


def julian_to_greg(JD):
    b = int(JD + 0.5) + 1537
    c = int((b - 122.1) / 365.25)
    d = int(365.25 * c)
    e = int((b - d) / 30.6001)

    hour = (JD + 0.5 - int(JD + 0.5)) * 24
    min = (hour - int(hour)) * 60
    hour = int(hour)
    sec = (min - int(min)) * 60
    min = int(min)
    day = b - d - int(30.6001 * e)
    month = e - 1 - 12 * int(e / 14)
    year = c - 4715 - int((month + 7) / 10)

    return (year, month, day, hour, min, sec)


def julian_to_year_frac(JD):
    year, month, day, hour, min, sec = julian_to_greg(JD)

    if year % 4 == 0:
        DAYS = {0: 0, 1: 31, 2: 60, 3: 91, 4: 121, 5: 152, 6: 182, 7: 213, 8: 244, 9: 274, 10: 305, 11: 335, 12: 366}
    else:
        DAYS = {0: 0, 1: 31, 2: 59, 3: 90, 4: 120, 5: 151, 6: 181, 7: 212, 8: 243, 9: 273, 10: 304, 11: 334, 12: 365}

    return year + (DAYS[month - 1] + day) / DAYS[12] + (hour + min / 60 + sec / 3600) / DAYS[12] / 24

def mjd(JD):
    return JD - 2400000.5


def jul_cent(JD):
    """
    Julian Centuries, starting from 2000 Jan 1d 12h, in UTC
    """

    return (JD - 2451545) / 36525



def JD_add_seconds(seconds):
    return seconds / 24 / 3600


def jul_date_ut1(JC, DUT1):
    """
    Julian date, starting from 2000 Jan 1d 12h, in UT1
    based on Julian Centuries in UTC
    """

    return 36525 * JC + DUT1 / 24 / 3600



def ERA(JC, DUT1):
    """
    Earth-Roation Angle in [rad]
    """

    return 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * jul_date_ut1(JC, DUT1))


def earth_rot(JC, DUT1):
    vartheta = ERA(JC, DUT1)

    return np.array([[np.cos(vartheta), -np.sin(vartheta), 0],
                     [np.sin(vartheta), np.cos(vartheta), 0],
                     [0, 0, 1]])


def tio_loc(JC):
    """
    TIO locator s' in [rad]
    """

    UARCSEC2RAD = np.pi / 180 / 3600 / 1000000

    return -47 * UARCSEC2RAD * JC

def TIP_coords(JD, table=TIP_AND_DUT1_TABLE):
    """
    returns:
            x_p in [rad]
            y_p in [rad]
            dUT1 in [sec]
    """

    ARCSEC2RAD = np.pi / 180 / 3600

    MJD = int(mjd(JD))

    if MJD > 59950 or MJD < 59586:
        raise Exception('Non-sustainable date, import another file')

    num = np.argwhere(table[:, 0] == MJD)[0, 0]

    x_p, y_p, DUT1 = table[num, 1:]

    return np.array([x_p * ARCSEC2RAD, y_p * ARCSEC2RAD, DUT1])

def polar_mot(x_p, y_p, JC):
    s = tio_loc(JC)

    R3 = np.array([[np.cos(s), -np.sin(s), 0],
                   [np.sin(s), np.cos(s), 0],
                   [0, 0, 1]])

    R2 = np.array([[np.cos(x_p), 0, -np.sin(x_p)],
                   [0, 1, 0],
                   [np.sin(x_p), 0, np.cos(x_p)]])

    R1 = np.array([[1, 0, 0],
                   [0, np.cos(y_p), np.sin(y_p)],
                   [0, -np.sin(y_p), np.cos(y_p)]])

    return R3 @ R2 @ R1

def DCM_ECEF_to_ECI(JD_UTC, tables_X=X_CIP_COEFS_TABLES, tables_Y=Y_CIP_COEFS_TABLES, tables_sXY=sXY_CIO_COEFS_TABLES,
                    table_TIP_and_DUT1=TIP_AND_DUT1_TABLE):
    JD_TT = JD_UTC + JD_add_seconds(37 + 32.184)
    JC_UTC = jul_cent(JD_UTC)
    JC_TT = jul_cent(JD_TT)
    x_p, y_p, DUT1 = TIP_coords(JD_UTC, table=table_TIP_and_DUT1)

    Q = nut_prec(JC_TT, tables_X=tables_X, tables_Y=tables_Y, tables_sXY=tables_sXY)
    R = earth_rot(JC_UTC, DUT1)
    W = polar_mot(x_p, y_p, JC_TT)

    return Q @ R @ W


def magn_field_ECI(JD_UTC, r, theta, phi, table_IGRF=IGRF_COEFS_TABLE, tables_X=X_CIP_COEFS_TABLES,
                   tables_Y=Y_CIP_COEFS_TABLES, tables_sXY=sXY_CIO_COEFS_TABLES, table_TIP_and_DUT1=TIP_AND_DUT1_TABLE):
    year = julian_to_year_frac(JD_UTC)
    B = magn_field_ECEF(year, r, theta, phi, table=table_IGRF)

    A = DCM_ECEF_to_ECI(JD_UTC, tables_X=tables_X, tables_Y=tables_Y, tables_sXY=tables_sXY,
                        table_TIP_and_DUT1=table_TIP_and_DUT1)

    return A @ B


'''
B = magn_field(2020, 6371200, np.pi/3, np.pi/2)

print('Magnetic fields components in South-East-Up geographical frame:', B, 'nT')
print('Intensity of the field:', np.linalg.norm(B), 'nT')
B_ECI = magn_field_ECI(2458940.1933954097, 6371200, np.pi / 3, np.pi / 2)

print('Magnetic fields components in ECI:', B_ECI, 'nT')
'''