"""Convenient trigonometric functions."""
import math

import numpy


def mag(vec):
    """Return the magnitude of a vector."""
    return numpy.linalg.norm(vec)


def deg2rad(ang):
    """Convert an angle from degrees to radians.

    Parameters
    ----------
    ang : float
       An angle, in degrees.

    Returns
    -------
    float
       The angle in radians.

    """
    return ang * numpy.pi / 180


def rad2deg(ang):
    """Convert an angle from radians to degrees.

    Parameters
    ----------
    ang : float
       An angle, in radians.

    Returns
    -------
    float
       The angle in degrees.

    """
    return ang * 180.0 / numpy.pi


def cosd(ang):
    """Return the cosine of an angle specified in degrees.

    Parameters
    ----------
    ang : float
       An angle, in degrees.

    """
    return numpy.cos(deg2rad(ang))


def sind(ang):
    """Return the sine of an angle specified in degrees.

    Parameters
    ----------
    ang : float
       An angle, in degrees.

    """
    return numpy.sin(deg2rad(ang))


def tand(ang):
    """Return the tangent of an angle specified in degrees.

    Parameters
    ----------
    ang : float
       An angle, in degrees.

    """
    return math.tan(deg2rad(ang))


def acosd(slope):
    """Return the arccosine of an angle specified in degrees.

    Returns
    -------
    float
       An angle, in degrees.

    """
    return rad2deg(math.acos(slope))


def asind(slope):
    """Return the arcsine of an angle specified in degrees.

    Returns
    -------
    float
       An angle, in degrees.

    """
    return rad2deg(math.asin(slope))


def atand(slope):
    """Return the arctangent of an angle specified in degrees.

    Returns
    -------
    float
       An angle, in degrees.

    """
    return rad2deg(math.atan(slope))


def atand2(delta_y, delta_x):
    """Return the arctan2 of an angle specified in degrees.

    Returns
    -------
    float
       An angle, in degrees.

    """
    return rad2deg(numpy.arctan2(delta_y, delta_x))


def ang2vec(ang):
    """Convert angle in radians to a 3-d unit vector in the x-y plane."""
    return numpy.array([numpy.cos(ang), numpy.sin(ang), 0.0])


def ang2vecd(ang):
    """Convert angle in degrees to a 3-d unit vector in the x-y plane."""
    return ang2vec(deg2rad(ang))


def angd2vec(ang):
    """Deprecated alias for ang2vecd."""
    return ang2vecd(ang)


def angd2vec2d(ang):
    """Convert angle in degrees to a 2-d unit vector in the x-y plane."""
    return ang2vecd(ang)[:2]


def quaternion_to_euler_angles(angle, x_comp, y_comp, z_comp):
    """Convert a quaternion to Euler angles.

    Parameters
    ----------
    angle : float
        The angle to rotate about the component vector.
    x_comp, y_comp, z_comp : float, float, float
        The x, y, and z components of the vector about which to rotate.

    """
    alpha = math.degrees(
        math.atan2(
            +2.0 * (angle * x_comp + y_comp * z_comp),
            +1.0 - 2.0 * (x_comp ** 2 + y_comp ** 2),
        )
    )

    beta = math.degrees(
        math.asin(max([min([+2.0 * (angle * y_comp - z_comp * x_comp), 1.0]), -1.0]))
    )

    gamma = math.degrees(
        math.atan2(
            +2.0 * (angle * z_comp + x_comp * y_comp),
            +1.0 - 2.0 * (y_comp ** 2 + z_comp ** 2),
        )
    )

    return alpha, beta, gamma


def rotate_vec_2d(vec, ang):
    """Rotate a 2d vector v by angle ang in degrees."""
    vec3d = numpy.zeros(3)
    vec3d[:2] = vec
    return rotate_vec(vec3d, 0, 0, ang)[:2]


def rotate_vec(vec, ang_x=0.0, ang_y=0.0, ang_z=0.0, about=numpy.zeros(3)):
    """Rotate a 3d vector v by angle ang in degrees."""
    rot_x = numpy.array(
        [[1, 0, 0], [0, cosd(ang_x), -sind(ang_x)], [0, sind(ang_x), cosd(ang_x)]]
    )
    rot_y = numpy.array(
        [[cosd(ang_y), 0, sind(ang_y)], [0, 1, 0], [-sind(ang_y), 0, cosd(ang_y)]]
    )
    rot_z = numpy.array(
        [[cosd(ang_z), -sind(ang_z), 0], [sind(ang_z), cosd(ang_z), 0], [0, 0, 1]]
    )

    return (
        numpy.dot(
            numpy.dot(numpy.dot(rot_x, rot_y), rot_z), (numpy.asarray(vec) - about).T
        )
        + about
    )


def cross2(vec_a, vec_b):
    """Return the cross product of two 2d vectors."""
    return vec_a[0] * vec_b[1] - vec_a[1] * vec_b[0]
