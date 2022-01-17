import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d import proj3d
import math_utils
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes


def Rotation2DPlot(time_hours, X_0, X, omg, Omgr):
    
    roll, pitch, yaw = math_utils.quat2rpy(X[0], X[1], X[2], X[3])
    
    f, ax = plt.subplots(4, sharex=True, figsize=(10,12))
    f.suptitle('Initial quaternion = [{0:.2f}, {1:.2f}, {2:.2f}, {3:.2f}]\nInitial angular velocity = [{4}, {5}, {6}]'
               .format(X_0[0], X_0[1], X_0[2], X_0[3], round(X_0[4], 3), round(X_0[5], 3), round(X_0[6], 3)), fontsize=16)
    f.set_figheight(9)
    f.set_figwidth(11)
    
    ax[0].plot(time_hours, omg[0, :], label='$\omega_1$')
    ax[0].plot(time_hours, omg[1, :], label='$\omega_2$')
    ax[0].plot(time_hours, omg[2, :], label='$\omega_3$')
    ax[0].set_title('Absolute angular velocity', fontsize=14)
    ax[0].set_ylabel('rad/s', fontsize=10)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    ax[0].grid()
    
    ax[1].plot(time_hours, Omgr[0, :], label='$\Omega_1$')
    ax[1].plot(time_hours, Omgr[1, :], label='$\Omega_2$')
    ax[1].plot(time_hours, Omgr[2, :], label='$\Omega_3$')
    ax[1].set_title('Relative angular velocity', fontsize=14)
    ax[1].set_ylabel('rad/s', fontsize=10)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    ax[1].grid()
    
    ax[2].plot(time_hours, roll/np.pi, label='Roll')
    ax[2].plot(time_hours, pitch/np.pi, label='Pitch')
    ax[2].plot(time_hours, yaw/np.pi, label='Yaw')
    ax[2].set_title('Orientation', fontsize=14)
    ax[2].yaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
    ax[2].set_ylabel('rad', fontsize=10)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    ax[2].grid()

    ax[3].plot(time_hours, X[1, :], label='$q_1$')
    ax[3].plot(time_hours, X[2, :], label='$q_2$')
    ax[3].plot(time_hours, X[3, :], label='$q_3$')
    ax[3].plot(time_hours, X[0, :], label='$q_0$')
    ax[3].set_title('Quaternion components', fontsize=14)
    ax[3].set_ylabel('units', fontsize=10)
    ax[3].set_xlabel('hours', fontsize=10)
    ax[3].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    ax[3].grid()
    
    plt.show()


def DynamicsPlot(time_hours, X_0, X, X_est, Omgr):
    roll_est, pitch_est, yaw_est = math_utils.quat2rpy_deg(X_est[0], X_est[1], X_est[2], X_est[3])
    offset = int(np.size(time_hours) * 2. / 3.)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=2, figsize=(16, 18))
    f.suptitle('Initial quaternion = [{0}, {1}, {2}, {3}]\nInitial angular velocity = [{4}, {5}, {6}] rad/s'
               .format(round(X_0[0], 3), round(X_0[1], 3), round(X_0[2], 3), round(X_0[3], 3),
                       round(X_0[4], 3), round(X_0[5], 3), round(X_0[6], 3)), fontsize=16)

    ax1[0].plot(time_hours, X[4], label='$\omega_1$')
    ax1[0].plot(time_hours, X[5], label='$\omega_2$')
    ax1[0].plot(time_hours, X[6], label='$\omega_3$')
    ax1[0].set_title('Absolute angular velocity', fontsize=14)
    ax1[0].set_ylabel('rad/s', fontsize=10)
    ax1[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    ax1[0].grid()

    ax1[1].plot(time_hours, Omgr[0, :], label='$\Omega_1$')
    ax1[1].plot(time_hours, Omgr[1, :], label='$\Omega_2$')
    ax1[1].plot(time_hours, Omgr[2, :], label='$\Omega_3$')
    ax1[1].set_title('Relative angular velocity', fontsize=14)
    ax1[1].set_ylabel('rad/s', fontsize=10)
    ax1[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    ax1[1].grid()


    ax2[0].plot(time_hours, X[1], label='$q_1$')
    ax2[0].plot(time_hours, X[2], label='$q_2$')
    ax2[0].plot(time_hours, X[3], label='$q_3$')
    ax2[0].plot(time_hours, X[0], label='$q_0$')
    ax2[0].set_title('Quaternion components', fontsize=14)
    # ax2[0].set_xlabel('hours', fontsize=10)
    ax2[0].set_ylabel('units', fontsize=10)
    ax2[0].grid()
    ax2[0].legend(loc='upper right')

    ax2[1].plot(time_hours, X_est[1], label='$q_1$')
    ax2[1].plot(time_hours, X_est[2], label='$q_2$')
    ax2[1].plot(time_hours, X_est[3], label='$q_3$')
    ax2[1].plot(time_hours, X_est[0], label='$q_0$')
    ax2[1].set_title('Quaternion estimated by EKF', fontsize=14)
    # ax2[1].set_xlabel('hours', fontsize=10)
    ax2[1].set_ylabel('units', fontsize=10)
    ax2[1].grid()
    ax2[1].legend(loc='upper right')

    ax3[0].plot(time_hours, roll_est, label='Roll')
    ax3[0].plot(time_hours, pitch_est, label='Pitch')
    ax3[0].plot(time_hours, yaw_est, label='Yaw')
    ax3[0].set_title('Orientation', fontsize=14)
    # ax3[0].set_xlabel('hours', fontsize=10)
    ax3[0].set_ylabel('[deg]', fontsize=10)
    ax3[0].grid()
    ax3[0].legend(loc='upper right')

    ax3[1].plot(time_hours[offset:], roll_est[offset:], label='Roll')
    ax3[1].plot(time_hours[offset:], pitch_est[offset:], label='Pitch')
    ax3[1].plot(time_hours[offset:], yaw_est[offset:], label='Yaw')
    ax3[1].set_title('Zoomed orientation', fontsize=14)
    # ax3[1].set_xlabel('hours', fontsize=10)
    ax3[1].set_ylabel('[deg]', fontsize=10)
    # ax3[1].set_xlim([time_hours[-1]/2, time_hours[-1]])
    ax3[1].grid()
    ax3[1].legend(loc='upper right')

    ax4[0].plot(time_hours, X_est[7])
    ax4[0].plot(time_hours, X_est[8])
    ax4[0].plot(time_hours, X_est[9])
    ax4[0].set_title('m_res_est', fontsize=14)
    ax4[0].set_xlabel('hours', fontsize=10)
    ax4[0].grid()
    # ax[5].set_ylabel('units', fontsize=10)

    ax4[1].plot(time_hours, X_est[10])
    ax4[1].plot(time_hours, X_est[11])
    ax4[1].plot(time_hours, X_est[12])
    ax4[1].set_title('B_bias_est', fontsize=14)
    ax4[1].set_xlabel('hours', fontsize=10)
    ax4[1].grid()

    plt.show()

def MagnetometerReadingsPlot(time_hours,B):
    
    plt.figure(figsize=(11,3))
    plt.plot(time_hours,B[0,:]*1e6,label="B1")
    plt.plot(time_hours,B[1,:]*1e6,label="B2")
    plt.plot(time_hours,B[2,:]*1e6,label="B3")
    plt.title('Magnetic field components', fontsize=14)
    plt.ylabel('microtesla', fontsize=10)
    plt.xlabel('hours', fontsize=10)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    plt.grid()
    
    plt.show()
    
def CoilPlot(time_hours,I,V):
    
    f, ax = plt.subplots(2, sharex=True, figsize=(10,12))
    f.set_figheight(6)
    f.set_figwidth(11)
    
    ax[0].plot(time_hours, I[0,:]*1e3, label="I1")
    ax[0].plot(time_hours, I[1,:]*1e3, label="I2")
    ax[0].plot(time_hours, I[2,:]*1e3, label="I3")
    ax[0].set_title('Magnetic coil current', fontsize=14)
    ax[0].set_ylabel('mAmp', fontsize=10)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    ax[0].grid()
    
    ax[1].plot(time_hours, V[0,:]*1e3, label="V1")
    ax[1].plot(time_hours, V[1,:]*1e3, label="V2")
    ax[1].plot(time_hours, V[2,:]*1e3, label="V3")
    ax[1].set_title('Magnetic coil control voltage', fontsize=14)
    ax[1].set_ylabel('mVolt', fontsize=10)
    ax[1].set_xlabel('hours', fontsize=10)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    ax[1].grid()
    
    plt.show()
    

def EKFComparisonPlot(time_hours,X_result,Q_est,omega):
    f, ax = plt.subplots(4, sharex=True, figsize=(10,10))
    
    ax[0].plot(time_hours, X_result[4,:], label='$\omega_1$')
    ax[0].plot(time_hours, X_result[5,:], label='$\omega_2$')
    ax[0].plot(time_hours, X_result[6,:], label='$\omega_3$')
    ax[0].set_title('Absolute angular velocity', fontsize=14)
    ax[0].grid()
    ax[0].set_ylabel('rad/s', fontsize=10)
    ax[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    
    ax[1].plot(time_hours, omega[0,:], label='$\omega_1$')
    ax[1].plot(time_hours, omega[1,:], label='$\omega_2$')
    ax[1].plot(time_hours, omega[2,:], label='$\omega_3$')
    ax[1].set_title('Absolute angular velocity estimated by EKF', fontsize=14)
    ax[1].grid()
    ax[1].set_ylabel('rad/s', fontsize=10)
    ax[1].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)

    ax[2].set_ylabel('[deg]', fontsize=10)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    ax[2].plot(time_hours, X_result[1,:], label='$q_1$')
    ax[2].plot(time_hours, X_result[2,:], label='$q_2$')
    ax[2].plot(time_hours, X_result[3,:], label='$q_3$')
    ax[2].plot(time_hours, X_result[0,:], label='$q_0$')
    ax[2].set_title('Quaternion components', fontsize=14)
    ax[2].grid()
    ax[2].set_ylabel('units', fontsize=10)
    ax[2].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    
    ax[3].plot(time_hours, Q_est[1,:], label='$q_1$')
    ax[3].plot(time_hours, Q_est[2,:], label='$q_2$')
    ax[3].plot(time_hours, Q_est[3,:], label='$q_3$')
    ax[3].plot(time_hours, Q_est[0,:], label='$q_0$')
    ax[3].set_title('Quaternion estimated by EKF', fontsize=14)
    ax[3].grid()
    ax[3].set_ylabel('units', fontsize=10)
    ax[3].set_xlabel('hours', fontsize=10)
    ax[3].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    
    
    plt.show()
    
    
def EKFConvergencePlot(time_hours,P):
    
    plt.figure(figsize=(11,3))
    plt.plot(time_hours,P[:])
    plt.title('EKF P matrix trace', fontsize=14)
    plt.ylabel('units', fontsize=10)
    plt.xlabel('hours', fontsize=10)
    plt.grid()
    
    plt.show()