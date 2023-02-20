from threading import *
import wx
import sys
import time
import numpy as np
import os
import pandas as pd
import scipy
import math
import re
import matplotlib.pyplot as plt


def text_alignment(x, y):
    """
    Align text labels based on the x- and y-axis coordinate values.

    This function is used for computing the appropriate alignment of the text
    label.

    For example, if the text is on the "right" side of the plot, we want it to
    be left-aligned. If the text is on the "top" side of the plot, we want it
    to be bottom-aligned.

    :param x, y: (`int` or `float`) x- and y-axis coordinate respectively.
    :returns: A 2-tuple of strings, the horizontal and vertical alignments
        respectively.
    """
    if x == 0:
        ha = "center"
    elif x > 0:
        ha = "left"
    else:
        ha = "right"
    if y == 0:
        va = "center"
    elif y > 0:
        va = "bottom"
    else:
        va = "top"

    return ha, va


def to_proper_radians(theta):
    """
    Converts theta (radians) to be within -pi and +pi.
    """
    if theta > np.pi or theta < -np.pi:
        theta = theta % np.pi
    return theta


def to_cartesian(r, theta, theta_units="radians"):
    """
    Converts polar r, theta to cartesian x, y.
    """
    assert theta_units in [
        "radians",
        "degrees",
    ], "kwarg theta_units must specified in radians or degrees"

    # Convert to radians
    # if theta_units == "degrees":
    #    theta = to_radians(theta)

    theta = to_proper_radians(theta)
    x = r * np.cos(theta)
    y = r * np.sin(theta)

    return x, y


def to_polar(x, y, theta_units="radians"):
    """
    Converts cartesian x, y to polar r, theta.
    """
    assert theta_units in [
        "radians",
        "degrees",
    ], "kwarg theta_units must specified in radians or degrees"

    theta = math.atan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)

    # if theta_units == "degrees":
    #    theta = to_degrees(to_proper_radians(theta) / np.pi * 180)

    return r, theta


def node_theta(nodelist, node):
    """
    Maps node to Angle.

    :param nodelist: Nodelist from the graph.
    :type nodelist: list.
    :param node: The node of interest. Must be in the nodelist.
    :returns: theta -- the angle of the node in radians.
    """
    assert len(nodelist) > 0, "nodelist must be a list of items."
    assert node in nodelist, "node must be inside nodelist."

    i = nodelist.index(node)
    theta = -np.pi + i * 2 * np.pi / len(nodelist)

    return theta


def B(i, N, t):
    val = scipy.special.comb(N, i) * t ** i * (1. - t) ** (N - i)
    return val


def P(t, X):
    '''
     xx = P(t, X)

     Evaluates a Bezier curve for the points in X.

     Inputs:
      X is a list (or array) or 2D coords: start , references , end
      t is a number (or list of numbers) in [0,1] where you want to
        evaluate the Bezier curve

     Output:
      xx is the set of 2D points along the Bezier curve
    '''
    X = np.array(X)
    N, d = np.shape(X)  # Number of points, Dimension of points
    N = N - 1
    xx = np.zeros((len(t), d))

    for i in range(N + 1):
        xx += np.outer(B(i, N, t), X[i])

    return xx
