import numpy as np


def normalize(obj):
    return obj / np.linalg.norm(obj)


def skew_symm(vec):
    if vec.ndim != 1:
        raise Exception("Not a vector")
    if len(vec) != 3:
        raise Exception("Wrong number of coordinates in vector: {}, should be 3".format(len(vec)))

    return np.array([[0., -vec[2], vec[1]], [vec[2], 0., -vec[0]], [-vec[1], vec[0], 0.]])


def cross_product(a, b):
    def check_dimensions(vec, string):

        if vec.ndim != 1:
            raise Exception("The {} input is not a vector".format(string))
        if len(vec) != 3:
            raise Exception("Wrong number of coordinates in the {0} vector: {1}, should be 3".format(string, len(vec)))

    check_dimensions(a, 'first')
    check_dimensions(b, 'second')

    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]])


def quat_product(q1, q2):
    def check_dimensions(quat, string):

        if quat.ndim != 1:
            raise Exception("The {} input is not a quaternion".format(string))
        if len(quat) != 4:
            raise Exception("Wrong number of elements in the {0} quaternion: {1}, must be 4".format(string, len(quat)))

    check_dimensions(q1, 'first')
    check_dimensions(q2, 'second')

    q = np.zeros(4)
    q[0] = q1[0] * q2[0] - q1[1:].dot(q2[1:])
    q[1:] = q1[0] * q2[1:] + q2[0] * q1[1:] + cross_product(q1[1:], q2[1:])

    return q


def rotate_vec_with_quat(q, vec):
    def check_dimensions(obj, is_quat):

        if obj.ndim != 1:
            raise Exception("Not a {}".format('quaternion' * is_quat + 'vector' * (1 - is_quat)))
        if len(obj) != (3 + 1 * is_quat):
            raise Exception("Wrong number of coordinates in the {0}: {1}, should be {2}"
                            .format('quaternion' * is_quat + 'vector' * (1 - is_quat), len(obj), 3 + 1 * is_quat))

    check_dimensions(q, True)
    check_dimensions(vec, False)

    q = quat_conjugate(q)

    qxvec = cross_product(q[1:], vec)

    return q[1:].dot(vec) * q[1:] + q[0] ** 2. * vec + 2. * q[0] * qxvec + cross_product(q[1:], qxvec)


def quat2rpy(q0, q1, q2, q3):
    roll = np.arctan2(2. * (q0 * q1 + q2 * q3), 1. - 2. * (q1 ** 2 + q2 ** 2))
    pitch = np.arcsin(2. * (q0 * q2 - q1 * q3))
    yaw = np.arctan2(2. * (q0 * q3 + q1 * q2), 1. - 2. * (q2 ** 2 + q3 ** 2))

    return [roll, pitch, yaw]

def quat2rpy_deg(q0, q1, q2, q3):
    roll = np.arctan2(2. * (q0 * q1 + q2 * q3), 1. - 2. * (q1 ** 2 + q2 ** 2)) * 180 / np.pi
    pitch = np.arcsin(2. * (q0 * q2 - q1 * q3)) * 180 / np.pi
    yaw = np.arctan2(2. * (q0 * q3 + q1 * q2), 1. - 2. * (q2 ** 2 + q3 ** 2)) * 180 / np.pi

    return [roll, pitch, yaw]

def quat_conjugate(q):
    q_new = np.copy(q)
    q_new[1:] *= -1.

    return q_new


def rotate_each_vector(array_of_vector, q):
    result = np.zeros(array_of_vector.shape)
    i = 0
    for vec in array_of_vector:
        result[i] = rotate_vec_with_quat(q, vec)
        i += 1
    return result


def matr2quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    q = np.zeros(4)

    if trace > 0.:
        s = 0.5 / np.sqrt(trace + 1)
        q[0] = 0.25 / s
        q[1] = (R[2, 1] - R[1, 2]) * s
        q[2] = (R[0, 2] - R[2, 0]) * s
        q[3] = (R[1, 0] - R[0, 1]) * s
    else:
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2. * np.sqrt(1. + R[0, 0] - R[1, 1] - R[2, 2])
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * np.sqrt(1. + R[1, 1] - R[0, 0] - R[2, 2])
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1][2] + R[2, 1]) / s
        else:
            s = 2. * np.sqrt(1. + R[2, 2] - R[0, 0] - R[1, 1])
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    return q
