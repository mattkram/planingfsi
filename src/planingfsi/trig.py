"""Convenient trigonometric functions."""
import math
from typing import Union

import numpy

ArrayLike = Union[float, numpy.ndarray]


def cosd(ang: ArrayLike) -> ArrayLike:
    """Return the cosine of an angle specified in degrees.

    Parameters
    ----------
    ang : float
       An angle, in degrees.

    """
    return numpy.cos(numpy.radians(ang))


def sind(ang: ArrayLike) -> ArrayLike:
    """Return the sine of an angle specified in degrees.

    Parameters
    ----------
    ang : float
       An angle, in degrees.

    """
    return numpy.sin(numpy.radians(ang))


def tand(ang: ArrayLike) -> ArrayLike:
    """Return the tangent of an angle specified in degrees.

    Parameters
    ----------
    ang : float
       An angle, in degrees.

    """
    return math.tan(numpy.radians(ang))


def acosd(slope: ArrayLike) -> ArrayLike:
    """Return the arccosine of an angle specified in degrees.

    Returns
    -------
    float
       An angle, in degrees.

    """
    return numpy.degrees(numpy.arccos(slope))


def asind(slope: ArrayLike) -> ArrayLike:
    """Return the arcsine of an angle specified in degrees.

    Returns
    -------
    float
       An angle, in degrees.

    """
    return numpy.degrees(numpy.arcsin(slope))


def atand(slope: ArrayLike) -> ArrayLike:
    """Return the arctangent of an angle specified in degrees.

    Returns
    -------
    float
       An angle, in degrees.

    """
    return numpy.degrees(numpy.arctan(slope))


def atand2(delta_y: ArrayLike, delta_x: ArrayLike) -> float:
    """Return the arctan2 of an angle specified in degrees.

    Returns
    -------
    float
       An angle, in degrees.

    """
    return float(numpy.degrees(numpy.arctan2(delta_y, delta_x)))


def ang2vec(ang: ArrayLike) -> numpy.ndarray:
    """Convert angle in radians to a 3-d unit vector in the x-y plane."""
    return numpy.array([numpy.cos(ang), numpy.sin(ang), 0.0])


def ang2vecd(ang: float) -> numpy.ndarray:
    """Convert angle in degrees to a 3-d unit vector in the x-y plane."""
    return ang2vec(numpy.radians(ang))


def angd2vec2d(ang: float) -> numpy.ndarray:
    """Convert angle in degrees to a 2-d unit vector in the x-y plane."""
    return ang2vecd(ang)[:2]


def rotate_vec_2d(vec: numpy.ndarray, ang: float) -> numpy.ndarray:
    """Rotate a 2d vector v by angle ang in degrees."""
    vec3d = numpy.zeros(3)
    vec3d[:2] = vec
    return rotate_vec(vec3d, 0, 0, ang)[:2]


def rotate_vec(
    vec: numpy.ndarray,
    ang_x: float = 0.0,
    ang_y: float = 0.0,
    ang_z: float = 0.0,
    about: numpy.ndarray = numpy.zeros(3),
) -> numpy.ndarray:
    """Rotate a 3d vector v by angle ang in degrees."""
    rot_x = numpy.array([[1, 0, 0], [0, cosd(ang_x), -sind(ang_x)], [0, sind(ang_x), cosd(ang_x)]])
    rot_y = numpy.array([[cosd(ang_y), 0, sind(ang_y)], [0, 1, 0], [-sind(ang_y), 0, cosd(ang_y)]])
    rot_z = numpy.array([[cosd(ang_z), -sind(ang_z), 0], [sind(ang_z), cosd(ang_z), 0], [0, 0, 1]])

    return (
        numpy.dot(numpy.dot(numpy.dot(rot_x, rot_y), rot_z), (numpy.asarray(vec) - about).T) + about
    )


def rotate_point(point: numpy.ndarray, about: numpy.ndarray, angle: float) -> numpy.ndarray:
    """Rotate a point about another point by a specific angle in degrees."""
    relative_pos = numpy.array(point) - numpy.array(about)
    new_pos = rotate_vec_2d(relative_pos, angle)
    return about + new_pos
