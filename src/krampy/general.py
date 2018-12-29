import errno
import os
import sys
from fnmatch import fnmatch

import numpy as np

"""General utilities."""
import errno
import os
import re
from fnmatch import fnmatch

import numpy


def mkdir_p(path):
    """Create directory if it doesn't exist"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def slugify(value):
    """Normalize a string, convert to lowercase, remove non-alpha characters, and
    convert spaces to hypens.

    Inspired by Django:
        https://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename.

    """
    value = re.sub(r"[^\w\s\.-]", "", value).strip()
    value = re.sub(r"[-\s]+", "-", value)
    return value


def camel_to_snake(camel_string):
    """Convert a CamelCase string to snake_string format."""
    first_cap_re = re.compile(r"(.)([A-Z][a-z]+)")
    all_cap_re = re.compile(r"([a-z0-9])([A-Z])")
    sub_1 = first_cap_re.sub(r"\1_\2", camel_string)
    return all_cap_re.sub(r"\1_\2", sub_1).lower()


def force_extension(filename, ext):
    """Force a file extension to be replaced in fileName"""
    # Remove existing extension
    split = filename.split(".")
    if len(split) > 1:
        filename = ".".join(split[:-1])

    return "".join((filename, ext))


def match(match_list, search_str, startindex=0, ignore=None, **kwargs):
    """Return the first index of an item in provided iterable where the search string is matched."""
    for i, val in enumerate(match_list[startindex:], start=startindex):
        if fnmatch(str(val), search_str) and i >= startindex:
            if val is not None and not val == ignore:
                return i
    if "default" in kwargs:
        return kwargs.get("default")
    raise ValueError("Search string {0} not found".format(search_str))


def index_match(index_list, match_list, search_str, **kwargs):
    """Index an array by matching a wildcard string in another list.

    Parameters
    ----------
    index_list : list of object
        The list to return a value from.
    match_list : list of str
        The list containing strings to match with the wildcard.
    searc_str : str
        A wildcard string to match.

    Returns
    -------
    object
        The first object in the index list that matches the wildcard.

    """
    index = match(match_list, search_str, **kwargs)
    if not isinstance(index, int):
        return kwargs.get("default")
    return index_list[index]


# def str2bool(str):
#     return str.lower() == 'true'
#
#
# def writeTimeHistoryFile(fileName, *args, **kwargs):
#     delimiter = kwargs.get('delimiter', '\t')
#     ff = open(fileName, 'w')
#     # Write header
#     ff.write('# ')
#     for arg in args:
#         if not arg[0] == '':
#             ff.write(arg[0])
#         if arg == args[-1]:
#             ff.write('\n')
#         else:
#             ff.write(delimiter)
#     # Write each row of values, and each arg is a different column
#     for i in range(len(args[0][1])):
#         for arg in args:
#             ff.write('{0:0.6e}'.format(arg[1][i]))
#             if arg == args[-1]:
#                 ff.write('\n')
#             else:
#                 ff.write(delimiter)
#     ff.close()
#
#
# def resampleTime(t, y, dt):
#     tNew = np.linspace(t[0], t[-1], int((t[-1] - t[0]) / dt))
#     yNew = np.zeros_like(tNew)
#     for i in range(len(tNew) - 1):
#         ind = np.nonzero(t <= tNew[i])[0][-1]
#
#         yNew[i] = y[ind] + (y[ind + 1] - y[ind]) * (tNew[i] - t[ind]) / (t[ind + 1] - t[ind])
#
#     return tNew, yNew
#
#
# def deriv(f, x):
#     '''
#     Calculate derivative of function f at points x using central differencing,
#     except at the ends where one-sided differencing is used.
#     '''
#     dfdx = np.zeros_like(x)
#     for i in range(len(x)):
#         if i == 0:
#             ind = [0, 1]
#         elif i == len(x) - 1:
#             ind = [-1, 0]
#         else:
#             ind = [-1, 1]
#         dfdx[i] = (f[i + ind[1]] - f[i + ind[0]]) / (x[i + ind[1]] - x[i + ind[0]])
#     return dfdx
#
#
# def mkdir_p(path):
#     '''Create directory if it doesn't exist'''
#     try:
#         os.makedirs(path)
#     except OSError as exc:
#         if exc.errno == errno.EEXIST and os.path.isdir(path):
#             pass
#         else:
#             raise
#
#
# def symlink_f(source, target):
#     '''Force-create symlink at target pointing to soirce'''
#     try:
#         os.symlink(source, target)
#     except OSError:
#         os.remove(target)
#         os.symlink(source, target)
#     except:
#         raise OSError('Failed to create symlink')
#
#
# def cumtrapz(x, y, reverse=False):
#     ''' Cumulative trapezoidal integration of a function.'''
#     if reverse:
#         x = x[::-1]
#         y = y[::-1]
#
#     I = np.zeros_like(x)
#     area = 0.5 * np.diff(x) * (y[1:] + y[:-1])
#     for i in range(1, len(I)):
#         I[i] = I[i - 1] + area[i - 1]
#
#     if reverse:
#         I = I[::-1]
#
#     return I
#
#
# def parseKeywordArgs(args=sys.argv[1:]):
#     return parseArgs(args)[1]
#
#
# def parseArgs(args=sys.argv[1:]):
#     '''Parse argument list, typically system arguments from calling script as executable.
#     '''
#     newargs = []
#     kwargs = {}
#     while len(args) > 0:
#         arg = args.pop(0)
#         if '=' in arg:
#             key, _, val = arg.partition('=')
#             if arg == 'True':
#                 kwargs[key] = True
#             elif val == 'False':
#                 kwargs[key] = False
#             else:
#                 try:
#                     kwargs[key] = float(val)
#                 except:
#                     kwargs[key] = val
#         else:
#             newargs.append(arg)
#     return tuple(newargs), kwargs

#
# def match(matchList, searchStr, startindex=0, ignore=None, **kwargs):
#     '''Return the first index of an item in provided iterable where the search string is matched.'''
#     for i, val in enumerate(matchList[startindex:], start=startindex):
#         if fnmatch(str(val), searchStr) and i >= startindex:
#             if val is not None and not val == ignore:
#                 return i
#     if 'default' in kwargs:
#         return kwargs.get('default')
#     raise ValueError('Search string {0} not found'.format(searchStr))
#
#
# def indexMatch(indexList, matchList, searchStr, **kwargs):
#     '''Index an array indexList by matching searchStr with the first matching element of searchList.'''
#     index = match(matchList, searchStr, **kwargs)
#     if not isinstance(index, int):
#         return kwargs.get('default')
#     return indexList[index]


# def trapz(x, f):
#     # Trapezoidal integration
#     I = np.argsort(x)
#     x = x[I]
#     f = f[I]
#     f[np.nonzero(np.abs(f) == float('Inf'))] = 0.0
#
#     return 0.5 * np.sum((x[1:] - x[0:-1]) * (f[1:] + f[0:-1]))
#
#
# def integrate(x, f):
#     return trapz(x, f)
#
#
# def growPoints(x0, x1, xMax, rate=1.1):
#     dx = x1 - x0
#     x = [x1]
#
#     if dx > 0:
#         done = lambda xt: xt > xMax
#     elif dx < 0:
#         done = lambda xt: xt < xMax
#     else:
#         done = lambda xt: True
#
#     while not done(x[-1]):
#         x.append(x[-1] + dx)
#         dx *= rate
#
#     return np.array(x[1:])
#
#
# def fillPoints(x0, x1, L, pctLast, targetRate=1.1):
#     dxLast = pctLast * L
#     x2 = x1 + np.sign(x1 - x0) * (L - 0.5 * dxLast)
#
#     func = lambda rate: growPoints(x0, x1, x2, rate)
#     res1 = lambda rate: np.abs(np.diff(func(rate)[-2:])[0]) - dxLast
#     res2 = lambda rate: func(rate)[-1] - x2
#
#     return func(fzero(res2, fzero(res1, targetRate)))
#
#
# def getDerivative(f, x, direction='c'):
#     dx = 1e-6
#     fr = f(x + dx)
#     fl = f(x - dx)
#
#     if direction[0].lower() == 'r' or np.isnan(fl):
#         return (fr - f(x)) / dx
#     elif direction[0].lower() == 'l' or np.isnan(fr):
#         return (f(x) - fl) / dx
#     else:
#         return (f(x + dx) - f(x - dx)) / (2 * dx)
#
#
# def rotatePt(oldPt, basePt, ang):
#     relPos = np.array(oldPt) - np.array(basePt)
#     newPos = rotateVec(relPos, ang)
#     newPt = basePt + newPos
#
#     return newPt
#
#
# def forceExtension(fileName, ext):
#     '''Force a file extension to be replaced in fileName'''
#     # Remove existing extension
#     split = fileName.split('.')
#     if len(split) > 1:
#         fileName = '.'.join(split[:-1])
#
#     return ''.join((fileName, ext))
#
#
# def getFiles(directory, pattern):
#     '''Function returns files in data directory matching provided wildcard pattern'''
#     return [os.path.join(directory, f) for f in os.listdir(directory) if fnmatch(f, pattern)]

# def extrap1d(interpolator):
#  xs = interpolator.x
#  ys = interpolator.y
#
#  def pointwise(x):
#    if x < xs[0]:
#      return ys[0]  + (x - xs[0]) * (ys[1] - ys[0]) / (xs[1] - xs[0])
#    elif x > xs[-1]:
#      return ys[-1] + (x - xs[-1]) * (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
#    else:
#      return interpolator(x)
#
#  return pointwise
#
# def minMax(x):
#  return min(x), max(x)
#
# def sign(x):
#  if x > 0:
#    return 1.0
#  elif x < 0:
#    return -1.0
#  else:
#    return 0.0
#
# def heaviside(x):
#  if x > 0:
#    return 1.0
#  elif x < 0:
#    return 0.0
#  else:
#    return 0.5
#
# def inRange(x, lims):
#  if x >= lims[0] and x <= lims[1]:
#    return True
#  else:
#    return False
#
# def createIfNotExist(dirName):
#  if not os.path.exists(dirName):
#    os.makedirs(dirName)#, 0755)
#
# def writeasdict(filename, *args, **kwargs):
#  dataFormat   = kwargs.get('dataFormat', '>+10.8e')
#  ff = open(filename, 'w')
#  for name, value in args:
#    ff.write('{2:{0}} : {3:{1}}\n'.format('<14', dataFormat, name, value))
#  ff.close()
#
# def writeaslist(filename, *args, **kwargs):
#  headerFormat = kwargs.get('headerFormat', '<15')
#  dataFormat   = kwargs.get('dataFormat', '>+10.8e')
#  ff = open(filename, 'w')
#  write(ff, headerFormat, [item for item in [arg[0] for arg in args]])
#  for value in zip(*[arg[1] for arg in args]):
#    write(ff, dataFormat, value)
#  ff.close()
#
# def write(ff, writeFormat, items):
#  if isinstance(items[0], str):
#    ff.write('# ')
#  else:
#    ff.write('  ')
#  ff.write(''.join('{1:{0}} '.format(writeFormat, item) for item in items) + '\n')
#
# def sortDirByNum(dirStr, direction='forward'):
#  num = np.array([float(''.join(i for i in dir.lower() if i.isdigit() or i == '.').strip('.')) for dir in dirStr])
#  if direction == 'reverse':
#    ind = np.argsort(num)[::-1]
#  else:
#    ind = np.argsort(num)
#  return [dirStr[i] for i in ind], num[ind]
#
# def getFG(x):
#  txMax = 5.0
#  N = 20
#
#  pt = np.array([-1., 0., 1.]) * np.sqrt(3./5.)
#  w  = np.array([5., 8., 5.]) / 9
#
#  t  = np.linspace(0.0, txMax / x, N + 1)
#  dt = 0.5 * (t[1] - t[0])
#
#  f = 0.
#  g = 0.
#  for i in range(N):
#    ti = t[i] + dt * (pt + 1)
#    F = w * dt * np.exp(-ti * x) / (ti**2 + 1)
#
#    f += np.sum(F)
#    g += np.sum(F * ti)
#
#  return f, g
#
# def ensureDict(Dict):
#  if isinstance(Dict, str):
#    Dict = Dictionary(Dict)
#  return Dict
#
# def rm_rf(dList):
#  dList = map(None, dList)
#  for d in dList:
#    for path in (os.path.join(d,f) for f in os.listdir(d)):
#      if os.path.isdir(path):
#        rm_rf(path)
#      else:
#        os.unlink(path)
#    os.rmdir(d)
#
# def checkDir(f):
#  if not os.path.isdir(f):
#    os.makedirs(f)
#
# def listdirNoHidden(d):
#  f = os.listdir(d)
#  rmInd = []
#  for i in range(len(f)):
#    if f[i].startswith('.'):
#      rmInd.append(i)
#  for i in range(len(rmInd)):
#    del f[rmInd[i]]
#  return f
