import numpy as onp
import jax.numpy as np
from jax.ops import index, index_add, index_update
from jax import lax
import math

def _raiseValueError(msg):
  raise ValueError(msg)

def identity_matrix():
  """Return 4x4 identity/unit matrix.

  >>> I = identity_matrix()
  >>> numpy.allclose(I, numpy.dot(I, I))
  True
  >>> numpy.sum(I), numpy.trace(I)
  (4.0, 4.0)
  >>> numpy.allclose(I, numpy.identity(4))
  True

  """
  return np.identity(4)
  

def translation_matrix(direction):
  """Return matrix to translate by direction vector.

  >>> v = numpy.random.random(3) - 0.5
  >>> numpy.allclose(v, translation_matrix(v)[:3, 3])
  True

  """
  M = np.identity(4)
  M = index_update(M, index[:3,3], direction[:3])
  return M


def translation_from_matrix(matrix):
  """Return translation vector from translation matrix.

  >>> v0 = numpy.random.random(3) - 0.5
  >>> v1 = translation_from_matrix(translation_matrix(v0))
  >>> numpy.allclose(v0, v1)
  True

  """
  return np.array(matrix, copy=False)[:3, 3].copy()


def reflection_matrix(point, normal):
  """Return matrix to mirror at plane defined by point and normal vector.

  >>> v0 = numpy.random.random(4) - 0.5
  >>> v0[3] = 1.
  >>> v1 = numpy.random.random(3) - 0.5
  >>> R = reflection_matrix(v0, v1)
  >>> numpy.allclose(2, numpy.trace(R))
  True
  >>> numpy.allclose(v0, numpy.dot(R, v0))
  True
  >>> v2 = v0.copy()
  >>> v2[:3] += v1
  >>> v3 = v0.copy()
  >>> v2[:3] -= v1
  >>> numpy.allclose(v2, numpy.dot(R, v3))
  True

  """
  normal = unit_vector(normal[:3])
  M = np.identity(4)
  M = index_add(M, index[:3,:3], -2.0*np.outer(normal,normal))
  M = index_update(M, index[:3,3], (2.0 * np.dot(point[:3], normal)) * normal)
  #M[:3, :3] -= 2.0 * numpy.outer(normal, normal)
  #M[:3, 3] = (2.0 * numpy.dot(point[:3], normal)) * normal
  return M


def reflection_from_matrix(matrix):
  """Return mirror plane point and normal vector from reflection matrix.

  >>> v0 = numpy.random.random(3) - 0.5
  >>> v1 = numpy.random.random(3) - 0.5
  >>> M0 = reflection_matrix(v0, v1)
  >>> point, normal = reflection_from_matrix(M0)
  >>> M1 = reflection_matrix(point, normal)
  >>> is_same_transform(M0, M1)
  True

  """
  M = np.array(matrix, dtype=np.float64, copy=False)
  # normal: unit eigenvector corresponding to eigenvalue -1
  w, V = np.linalg.eig(M[:3, :3])
  i = np.where(abs(np.real(w) + 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no unit eigenvector corresponding to eigenvalue -1')
  normal = np.real(V[:, i[0]]).squeeze()
  # point: any unit eigenvector corresponding to eigenvalue 1
  w, V = np.linalg.eig(M)
  i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
  point = np.real(V[:, i[-1]]).squeeze()
  point /= point[3]
  return point, normal


def rotation_matrix(angle, direction, point=None):
  """Return matrix to rotate about axis defined by point and direction.

  >>> R = rotation_matrix(math.pi/2, [0, 0, 1], [1, 0, 0])
  >>> numpy.allclose(numpy.dot(R, [0, 0, 0, 1]), [1, -1, 0, 1])
  True
  >>> angle = (random.random() - 0.5) * (2*math.pi)
  >>> direc = numpy.random.random(3) - 0.5
  >>> point = numpy.random.random(3) - 0.5
  >>> R0 = rotation_matrix(angle, direc, point)
  >>> R1 = rotation_matrix(angle-2*math.pi, direc, point)
  >>> is_same_transform(R0, R1)
  True
  >>> R0 = rotation_matrix(angle, direc, point)
  >>> R1 = rotation_matrix(-angle, -direc, point)
  >>> is_same_transform(R0, R1)
  True
  >>> I = numpy.identity(4, numpy.float64)
  >>> numpy.allclose(I, rotation_matrix(math.pi*2, direc))
  True
  >>> numpy.allclose(2, numpy.trace(rotation_matrix(math.pi/2,
  ...                                               direc, point)))
  True

  """
  sina = np.sin(angle)
  cosa = np.cos(angle)
  direction = unit_vector(direction[:3])
  # rotation matrix around unit vector
  R = np.diag(np.array([cosa, cosa, cosa]))
  R = R + np.outer(direction, direction) * (1.0 - cosa)
  direction = direction * sina
  R = R + np.array([[ 0.0,         -direction[2],  direction[1]],
                    [ direction[2], 0.0,          -direction[0]],
                    [-direction[1], direction[0],  0.0]])
  M = np.identity(4)
  M = index_update(M, index[:3, :3], R)
  if point is not None:
    # rotation not around origin
    point = np.array(point[:3], dtype=np.float64, copy=False)
    M = index_update(M, index[:3, 3], point - np.dot(R, point))
  return M


def rotation_from_matrix(matrix):
  """Return rotation angle and axis from rotation matrix.

  >>> angle = (random.random() - 0.5) * (2*math.pi)
  >>> direc = numpy.random.random(3) - 0.5
  >>> point = numpy.random.random(3) - 0.5
  >>> R0 = rotation_matrix(angle, direc, point)
  >>> angle, direc, point = rotation_from_matrix(R0)
  >>> R1 = rotation_matrix(angle, direc, point)
  >>> is_same_transform(R0, R1)
  True

  """
  R = np.array(matrix, dtype=np.float64, copy=False)
  R33 = R[:3, :3]
  # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
  w, W = np.linalg.eig(np.transpose(R33))
  i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
  direction = np.real(W[:, i[-1]]).squeeze()
  # point: unit eigenvector of R33 corresponding to eigenvalue of 1
  w, Q = np.linalg.eig(R)
  i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
  point = np.real(Q[:, i[-1]]).squeeze()
  point /= point[3]
  # rotation angle depending on direction
  cosa = (np.trace(R33) - 1.0) / 2.0
  if abs(direction[2]) > 1e-8:
      sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
  elif abs(direction[1]) > 1e-8:
      sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
  else:
      sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
  angle = np.arctan2(sina, cosa)
  return angle, direction, point


def scale_matrix(factor, origin=None, direction=None):
  """Return matrix to scale by factor around origin in direction.

  Use factor -1 for point symmetry.

  >>> v = (numpy.random.rand(4, 5) - 0.5) * 20
  >>> v[3] = 1
  >>> S = scale_matrix(-1.234)
  >>> numpy.allclose(numpy.dot(S, v)[:3], -1.234*v[:3])
  True
  >>> factor = random.random() * 10 - 5
  >>> origin = numpy.random.random(3) - 0.5
  >>> direct = numpy.random.random(3) - 0.5
  >>> S = scale_matrix(factor, origin)
  >>> S = scale_matrix(factor, origin, direct)

  """
  if direction is None:
    # uniform scaling
    M = np.diag(np.array([factor, factor, factor, 1.0]))
    if origin is not None:
      #COMMENT(@cpgoodri): these should probably be combined...
      M = index_update(M, index[:3, 3], origin[:3])
      M = index_update(M, index[:3, 3], M[:3, 3] * (1.0 - factor))
      #M[:3, 3] = origin[:3]
      #M[:3, 3] *= 1.0 - factor
  else:
    # nonuniform scaling
    direction = unit_vector(direction[:3])
    factor = 1.0 - factor
    M = np.identity(4)
    M = index_add(M, index[:3, :3], -factor * np.outer(direction, direction))
    #M[:3, :3] -= factor * np.outer(direction, direction)
    if origin is not None:
      M = index_update(M, index[:3, 3], (factor * np.dot(origin[:3], direction)) * direction)
      #M[:3, 3] = (factor * numpy.dot(origin[:3], direction)) * direction
  return M


def scale_from_matrix(matrix):
  """Return scaling factor, origin and direction from scaling matrix.

  >>> factor = random.random() * 10 - 5
  >>> origin = numpy.random.random(3) - 0.5
  >>> direct = numpy.random.random(3) - 0.5
  >>> S0 = scale_matrix(factor, origin)
  >>> factor, origin, direction = scale_from_matrix(S0)
  >>> S1 = scale_matrix(factor, origin, direction)
  >>> is_same_transform(S0, S1)
  True
  >>> S0 = scale_matrix(factor, origin, direct)
  >>> factor, origin, direction = scale_from_matrix(S0)
  >>> S1 = scale_matrix(factor, origin, direction)
  >>> is_same_transform(S0, S1)
  True

  """
  M = np.array(matrix, dtype=np.float64, copy=False)
  M33 = M[:3, :3]
  factor = np.trace(M33) - 2.0
  try:
    # direction: unit eigenvector corresponding to eigenvalue factor
    w, V = np.linalg.eig(M33)
    i = np.where(abs(np.real(w) - factor) < 1e-8)[0][0]
    direction = np.real(V[:, i]).squeeze()
    direction /= vector_norm(direction)
  
  #WARNING(@cpgoodri): I'm not sure if this error-handling approach works with JAX, but it seems to pass tests...
  except IndexError:
    # uniform scaling
    factor = (factor + 2.0) / 3.0
    direction = None
  # origin: any eigenvector corresponding to eigenvalue 1
  w, V = np.linalg.eig(M)
  i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no eigenvector corresponding to eigenvalue 1')
  origin = np.real(V[:, i[-1]]).squeeze()
  origin /= origin[3]
  return factor, origin, direction


def projection_matrix(point, normal, direction=None,
                    perspective=None, pseudo=False):
  """Return matrix to project onto plane defined by point and normal.

  Using either perspective point, projection direction, or none of both.

  If pseudo is True, perspective projections will preserve relative depth
  such that Perspective = dot(Orthogonal, PseudoPerspective).

  >>> P = projection_matrix([0, 0, 0], [1, 0, 0])
  >>> numpy.allclose(P[1:, 1:], numpy.identity(4)[1:, 1:])
  True
  >>> point = numpy.random.random(3) - 0.5
  >>> normal = numpy.random.random(3) - 0.5
  >>> direct = numpy.random.random(3) - 0.5
  >>> persp = numpy.random.random(3) - 0.5
  >>> P0 = projection_matrix(point, normal)
  >>> P1 = projection_matrix(point, normal, direction=direct)
  >>> P2 = projection_matrix(point, normal, perspective=persp)
  >>> P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
  >>> is_same_transform(P2, numpy.dot(P0, P3))
  True
  >>> P = projection_matrix([3, 0, 0], [1, 1, 0], [1, 0, 0])
  >>> v0 = (numpy.random.rand(4, 5) - 0.5) * 20
  >>> v0[3] = 1
  >>> v1 = numpy.dot(P, v0)
  >>> numpy.allclose(v1[1], v0[1])
  True
  >>> numpy.allclose(v1[0], 3-v1[1])
  True

  """
  M = np.identity(4)
  point = np.array(point[:3], dtype=np.float64, copy=False)
  normal = unit_vector(normal[:3])
  if perspective is not None:
    # perspective projection
    perspective = np.array(perspective[:3], dtype=np.float64, copy=False)
    temp = np.dot(perspective-point, normal)
    M = index_update(M, index[0, 0], temp)
    M = index_update(M, index[1, 1], temp)
    M = index_update(M, index[2, 2], temp)
    M = index_add(M, index[:3, :3], -np.outer(perspective, normal))
    if pseudo:
      # preserve relative depth
      M = index_add(M, index[:3, :3], -np.outer(normal, normal))
      M = index_update(M, index[:3, 3], np.dot(point, normal) * (perspective+normal))
    else:
      M = index_update(M, index[:3, 3], np.dot(point, normal) * perspective)
    M = index_update(M, index[3, :3], -normal)
    M = index_update(M, index[3, 3], np.dot(perspective, normal))
  elif direction is not None:
    # parallel projection
    direction = np.array(direction[:3], dtype=np.float64, copy=False)
    scale = np.dot(direction, normal)
    M = index_add(M, index[:3, :3], -np.outer(direction, normal) / scale)
    M = index_update(M, index[:3, 3], direction * (np.dot(point, normal) / scale))
  else:
    # orthogonal projection
    M = index_add(M, index[:3, :3], -np.outer(normal, normal))
    M = index_update(M, index[:3, 3], np.dot(point, normal) * normal)
  return M


def projection_from_matrix(matrix, pseudo=False):
  """Return projection plane and perspective point from projection matrix.

  Return values are same as arguments for projection_matrix function:
  point, normal, direction, perspective, and pseudo.

  >>> point = numpy.random.random(3) - 0.5
  >>> normal = numpy.random.random(3) - 0.5
  >>> direct = numpy.random.random(3) - 0.5
  >>> persp = numpy.random.random(3) - 0.5
  >>> P0 = projection_matrix(point, normal)
  >>> result = projection_from_matrix(P0)
  >>> P1 = projection_matrix(*result)
  >>> is_same_transform(P0, P1)
  True
  >>> P0 = projection_matrix(point, normal, direct)
  >>> result = projection_from_matrix(P0)
  >>> P1 = projection_matrix(*result)
  >>> is_same_transform(P0, P1)
  True
  >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
  >>> result = projection_from_matrix(P0, pseudo=False)
  >>> P1 = projection_matrix(*result)
  >>> is_same_transform(P0, P1)
  True
  >>> P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
  >>> result = projection_from_matrix(P0, pseudo=True)
  >>> P1 = projection_matrix(*result)
  >>> is_same_transform(P0, P1)
  True

  """
  M = np.array(matrix, dtype=np.float64, copy=False)
  M33 = M[:3, :3]
  w, V = np.linalg.eig(M)
  i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
  if not pseudo and len(i):
    # point: any eigenvector corresponding to eigenvalue 1
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    # direction: unit eigenvector corresponding to eigenvalue 0
    w, V = np.linalg.eig(M33)
    i = np.where(abs(np.real(w)) < 1e-8)[0]
    if not len(i):
      raise ValueError('no eigenvector corresponding to eigenvalue 0')
    direction = np.real(V[:, i[0]]).squeeze()
    direction /= vector_norm(direction)
    # normal: unit eigenvector of M33.T corresponding to eigenvalue 0
    w, V = np.linalg.eig(M33.T)
    i = np.where(abs(np.real(w)) < 1e-8)[0]
    if len(i):
      # parallel projection
      normal = np.real(V[:, i[0]]).squeeze()
      normal /= vector_norm(normal)
      return point, normal, direction, None, False
    else:
      # orthogonal projection, where normal equals direction vector
      return point, direction, None, None, False
  else:
    # perspective projection
    i = np.where(abs(np.real(w)) > 1e-8)[0]
    if not len(i):
      raise ValueError(
          'no eigenvector not corresponding to eigenvalue 0')
    point = np.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    normal = - M[3, :3]
    perspective = M[:3, 3] / np.dot(point[:3], normal)
    if pseudo:
      perspective -= normal
    return point, normal, None, perspective, pseudo


def clip_matrix(left, right, bottom, top, near, far, perspective=False):
  print("WARNING: not implemented.")
  raise NotImplementedError



def shear_matrix(angle, direction, point, normal):
  """Return matrix to shear by angle along direction vector on shear plane.

  The shear plane is defined by a point and normal vector. The direction
  vector must be orthogonal to the plane's normal vector.

  A point P is transformed by the shear matrix into P" such that
  the vector P-P" is parallel to the direction vector and its extent is
  given by the angle of P-P'-P", where P' is the orthogonal projection
  of P onto the shear plane.

  >>> angle = (random.random() - 0.5) * 4*math.pi
  >>> direct = numpy.random.random(3) - 0.5
  >>> point = numpy.random.random(3) - 0.5
  >>> normal = numpy.cross(direct, numpy.random.random(3))
  >>> S = shear_matrix(angle, direct, point, normal)
  >>> numpy.allclose(1, numpy.linalg.det(S))
  True

  """
  normal = unit_vector(normal[:3])
  direction = unit_vector(direction[:3])
  
  #lax.cond(np.abs(np.dot(normal, direction)) > 1e-6,
  #    'direction and normal vectors are not orthogonal', _raiseValueError,
  #    None, lambda x: None)
  if np.abs(np.dot(normal, direction)) > 1e-6:
    raise ValueError('direction and normal vectors are not orthogonal')
  angle = np.tan(angle)
  M = np.identity(4)
  M = index_add(M, index[:3, :3], angle * np.outer(direction, normal))
  #M[:3, :3] += angle * numpy.outer(direction, normal)
  M = index_update(M, index[:3, 3], -angle * np.dot(point[:3], normal) * direction)
  #M[:3, 3] = -angle * numpy.dot(point[:3], normal) * direction
  return M


def shear_from_matrix(matrix):
  """Return shear angle, direction and plane from shear matrix.

  >>> angle = (random.random() - 0.5) * 4*math.pi
  >>> direct = numpy.random.random(3) - 0.5
  >>> point = numpy.random.random(3) - 0.5
  >>> normal = numpy.cross(direct, numpy.random.random(3))
  >>> S0 = shear_matrix(angle, direct, point, normal)
  >>> angle, direct, point, normal = shear_from_matrix(S0)
  >>> S1 = shear_matrix(angle, direct, point, normal)
  >>> is_same_transform(S0, S1)
  True

  """
  M = np.array(matrix, dtype=np.float64, copy=False)
  M33 = M[:3, :3]
  # normal: cross independent eigenvectors corresponding to the eigenvalue 1
  w, V = np.linalg.eig(M33)
  i = np.where(abs(np.real(w) - 1.0) < 1e-4)[0]
  if len(i) < 2:
    raise ValueError('no two linear independent eigenvectors found %s' % w)
  V = np.real(V[:, i]).squeeze().T
  lenorm = -1.0
  for i0, i1 in ((0, 1), (0, 2), (1, 2)):
    n = np.cross(V[i0], V[i1])
    w = vector_norm(n)
    if w > lenorm:
      lenorm = w
      normal = n
  normal /= lenorm
  # direction and angle
  direction = np.dot(M33 - np.identity(3), normal)
  angle = vector_norm(direction)
  direction /= angle
  angle = np.arctan(angle)
  # point: eigenvector corresponding to eigenvalue 1
  w, V = np.linalg.eig(M)
  i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no eigenvector corresponding to eigenvalue 1')
  point = np.real(V[:, i[-1]]).squeeze()
  point /= point[3]
  return angle, direction, point, normal


def decompose_matrix(matrix):
  print("WARNING: not implemented.")
  raise NotImplementedError


def compose_matrix(scale=None, shear=None, angles=None, translate=None,
                   perspective=None):
  print("WARNING: not implemented.")
  raise NotImplementedError


def orthogonalization_matrix(lengths, angles):
  print("WARNING: not implemented.")
  raise NotImplementedError


def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
  print("WARNING: not implemented.")
  raise NotImplementedError


def superimposition_matrix(v0, v1, scale=False, usesvd=True):
  print("WARNING: not implemented.")
  raise NotImplementedError


def euler_matrix(ai, aj, ak, axes='sxyz'):
  """Return homogeneous rotation matrix from Euler angles and axis sequence.

  ai, aj, ak : Euler's roll, pitch and yaw angles
  axes : One of 24 axis sequences as string or encoded tuple

  >>> R = euler_matrix(1, 2, 3, 'syxz')
  >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
  True
  >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
  >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
  True
  >>> ai, aj, ak = (4*math.pi) * (numpy.random.random(3) - 0.5)
  >>> for axes in _AXES2TUPLE.keys():
  ...    R = euler_matrix(ai, aj, ak, axes)
  >>> for axes in _TUPLE2AXES.keys():
  ...    R = euler_matrix(ai, aj, ak, axes)

  """
  try:
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
  except (AttributeError, KeyError):
    _TUPLE2AXES[axes]  # noqa: validation
    firstaxis, parity, repetition, frame = axes

  i = firstaxis
  j = _NEXT_AXIS[i+parity]
  k = _NEXT_AXIS[i-parity+1]

  if frame:
    ai, ak = ak, ai
  if parity:
    ai, aj, ak = -ai, -aj, -ak

  si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
  ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
  cc, cs = ci*ck, ci*sk
  sc, ss = si*ck, si*sk

  M = np.identity(4)
  if repetition:
    M = index_update(M, index[i, i], cj)
    #M[i, i] = cj
    M = index_update(M, index[i, j], sj*si)
    #M[i, j] = sj*si
    M = index_update(M, index[i, k], sj*ci)
    #M[i, k] = sj*ci
    M = index_update(M, index[j, i], sj*sk)
    #M[j, i] = sj*sk
    M = index_update(M, index[j, j], -cj*ss+cc)
    #M[j, j] = -cj*ss+cc
    M = index_update(M, index[j, k], -cj*cs-sc)
    #M[j, k] = -cj*cs-sc
    M = index_update(M, index[k, i], -sj*ck)
    #M[k, i] = -sj*ck
    M = index_update(M, index[k, j], cj*sc+cs)
    #M[k, j] = cj*sc+cs
    M = index_update(M, index[k, k], cj*cc-ss)
    #M[k, k] = cj*cc-ss
  else:
    M = index_update(M, index[i, i], cj*ck)
    #M[i, i] = cj*ck
    M = index_update(M, index[i, j], sj*sc-cs)
    #M[i, j] = sj*sc-cs
    M = index_update(M, index[i, k], sj*cc+ss)
    #M[i, k] = sj*cc+ss
    M = index_update(M, index[j, i], cj*sk)
    #M[j, i] = cj*sk
    M = index_update(M, index[j, j], sj*ss+cc)
    #M[j, j] = sj*ss+cc
    M = index_update(M, index[j, k], sj*cs-sc)
    #M[j, k] = sj*cs-sc
    M = index_update(M, index[k, i], -sj)
    #M[k, i] = -sj
    M = index_update(M, index[k, j], cj*si)
    #M[k, j] = cj*si
    M = index_update(M, index[k, k], cj*ci)
    #M[k, k] = cj*ci
  return M


def euler_from_matrix(matrix, axes='sxyz'):
  """Return Euler angles from rotation matrix for specified axis sequence.

  axes : One of 24 axis sequences as string or encoded tuple

  Note that many Euler angle triplets can describe one matrix.

  >>> R0 = euler_matrix(1, 2, 3, 'syxz')
  >>> al, be, ga = euler_from_matrix(R0, 'syxz')
  >>> R1 = euler_matrix(al, be, ga, 'syxz')
  >>> numpy.allclose(R0, R1)
  True
  >>> angles = (4*math.pi) * (numpy.random.random(3) - 0.5)
  >>> for axes in _AXES2TUPLE.keys():
  ...    R0 = euler_matrix(axes=axes, *angles)
  ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
  ...    if not numpy.allclose(R0, R1): print(axes, "failed")

  """
  try:
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
  except (AttributeError, KeyError):
    _TUPLE2AXES[axes]  # noqa: validation
    firstaxis, parity, repetition, frame = axes

  i = firstaxis
  j = _NEXT_AXIS[i+parity]
  k = _NEXT_AXIS[i-parity+1]

  M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
  if repetition:
    sy = np.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
    ax, ay, az = lax.cond(sy > _EPS,
        (np.arctan2(M[i, j],M[i, k]), np.arctan2(sy,M[i, i]), np.arctan2(M[j, i],-M[k, i])), lambda x: x,
        (np.arctan2(-M[j, k],M[j, j]), np.arctan2(sy,M[i, i]), 0.0), lambda x: x)

    """
    if sy > _EPS:
      ax = np.arctan2( M[i, j],  M[i, k])
      ay = np.arctan2( sy,       M[i, i])
      az = np.arctan2( M[j, i], -M[k, i])
    else:
      ax = np.arctan2(-M[j, k],  M[j, j])
      ay = np.arctan2( sy,       M[i, i])
      az = 0.0
    """
  else:
    cy = np.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
    ax, ay, az = lax.cond( cy > _EPS,
        (np.arctan2(M[k, j], M[k, k]), np.arctan2(-M[k, i], cy), np.arctan2(M[j, i], M[i, i])), lambda x: x,
        (np.arctan2(-M[j, k], M[j, j]), np.arctan2(-M[k, i], cy), 0.0), lambda x: x)

    """
    if cy > _EPS:
      ax = np.arctan2( M[k, j],  M[k, k])
      ay = np.arctan2(-M[k, i],  cy)
      az = np.arctan2( M[j, i],  M[i, i])
    else:
      ax = np.arctan2(-M[j, k],  M[j, j])
      ay = np.arctan2(-M[k, i],  cy)
      az = 0.0
    """

  if parity:
    ax, ay, az = -ax, -ay, -az
  if frame:
    ax, az = az, ax
  return ax, ay, az


def euler_from_quaternion(quaternion, axes='sxyz'):
  """Return Euler angles from quaternion for specified axis sequence.

  >>> angles = euler_from_quaternion([0.99810947, 0.06146124, 0, 0])
  >>> numpy.allclose(angles, [0.123, 0, 0])
  True

  """
  return euler_from_matrix(quaternion_matrix(quaternion), axes)


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
  """Return quaternion from Euler angles and axis sequence.

  ai, aj, ak : Euler's roll, pitch and yaw angles
  axes : One of 24 axis sequences as string or encoded tuple

  >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
  >>> numpy.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
  True

  """
  try:
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
  except (AttributeError, KeyError):
    _TUPLE2AXES[axes]  # noqa: validation
    firstaxis, parity, repetition, frame = axes

  i = firstaxis + 1
  j = _NEXT_AXIS[i+parity-1] + 1
  k = _NEXT_AXIS[i-parity] + 1

  if frame:
    ai, ak = ak, ai
  if parity:
    aj = -aj

  ai /= 2.0
  aj /= 2.0
  ak /= 2.0
  ci = np.cos(ai)
  si = np.sin(ai)
  cj = np.cos(aj)
  sj = np.sin(aj)
  ck = np.cos(ak)
  sk = np.sin(ak)
  cc = ci*ck
  cs = ci*sk
  sc = si*ck
  ss = si*sk

  q = np.empty((4, ))
  if repetition:
    q = index_update(q, index[0], cj*(cc - ss))
    #q[0] = cj*(cc - ss)
    q = index_update(q, index[i], cj*(cs + sc))
    #q[i] = cj*(cs + sc)
    q = index_update(q, index[j], sj*(cc + ss))
    #q[j] = sj*(cc + ss)
    q = index_update(q, index[k], sj*(cs - sc))
    #q[k] = sj*(cs - sc)
  else:
    q = index_update(q, index[0], cj*cc + sj*ss)
    #q[0] = cj*cc + sj*ss
    q = index_update(q, index[i], cj*sc - sj*cs)
    #q[i] = cj*sc - sj*cs
    q = index_update(q, index[j], cj*ss + sj*cc)
    #q[j] = cj*ss + sj*cc
    q = index_update(q, index[k], cj*cs - sj*sc)
    #q[k] = cj*cs - sj*sc
  if parity:
    q = index_update(q, index[j], -1.0 * q[j])
    #q[j] *= -1.0

  return q


def quaternion_about_axis(angle, axis):
  """Return quaternion for rotation about axis.

  >>> q = quaternion_about_axis(0.123, [1, 0, 0])
  >>> numpy.allclose(q, [0.99810947, 0.06146124, 0, 0])
  True

  """
  q = np.array([0.0, axis[0], axis[1], axis[2]])
  qlen = vector_norm(q)
  q = lax.cond(qlen > _EPS,
      q * np.sin(angle/2.0) / qlen, lambda x: x,
      q, lambda x: x)
  #if qlen > _EPS:
  #  q *= math.sin(angle/2.0) / qlen
  q = index_update(q, index[0], np.cos(angle/2.0))
  #q[0] = math.cos(angle/2.0)
  return q

















def quaternion_matrix(quaternion):
  """Return homogeneous rotation matrix from quaternion.

  >>> M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
  >>> numpy.allclose(M, rotation_matrix(0.123, [1, 0, 0]))
  True
  >>> M = quaternion_matrix([1, 0, 0, 0])
  >>> numpy.allclose(M, numpy.identity(4))
  True
  >>> M = quaternion_matrix([0, 1, 0, 0])
  >>> numpy.allclose(M, numpy.diag([1, -1, -1, 1]))
  True

  """
  q = np.array(quaternion, dtype=np.float64, copy=True)
  n = np.dot(q, q)

  def calc_mat_posn(qn):
    q, n = qn
    q *= np.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

  return lax.cond( n < _EPS,
      np.identity(4), lambda x: x,
      (q,n), calc_mat_posn)

  """
  if n < _EPS:
      return np.identity(4)
  q *= np.sqrt(2.0 / n)
  q = np.outer(q, q)
  return np.array([
      [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
      [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
      [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
      [                0.0,                 0.0,                 0.0, 1.0]])
  """













#############################################
################### Helpers #################
#############################################

def random_quaternion(rand=None):
  """Return uniform random unit quaternion.

  rand: array like or None
      Three independent random variables that are uniformly distributed
      between 0 and 1.

  >>> q = random_quaternion()
  >>> numpy.allclose(1, vector_norm(q))
  True
  >>> q = random_quaternion(numpy.random.random(3))
  >>> len(q.shape), q.shape[0]==4
  (1, True)

  """
  #TODO(cpgoodri): make this fully JAX-compatible
  #Currently, it is jax-compatible only if you pass rand manually
  if rand is None:
    rand = np.array(onp.random.rand(3))
  else:
    rand = np.array(rand, dtype=np.float64)
    assert len(rand) == 3
  r1 = np.sqrt(1.0 - rand[0])
  r2 = np.sqrt(rand[0])
  pi2 = np.pi * 2.0
  t1 = pi2 * rand[1]
  t2 = pi2 * rand[2]
  return np.array([np.cos(t2)*r2, np.sin(t1)*r1,
                      np.cos(t1)*r1, np.sin(t2)*r2])


def random_rotation_matrix(rand=None):
  """Return uniform random rotation matrix.

  rand: array like
      Three independent random variables that are uniformly distributed
      between 0 and 1 for each returned quaternion.

  >>> R = random_rotation_matrix()
  >>> numpy.allclose(numpy.dot(R.T, R), numpy.identity(4))
  True

  """
  return quaternion_matrix(random_quaternion(rand))


# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())









def vector_norm(data, axis=None, out=None):
  """Return length, i.e. Euclidean norm, of ndarray along axis.

  COMMENT(@cpgoodri): For now, I am only implementing this for out=None. I don't *think* this will affect internal functionality.

  >>> v = numpy.random.random(3)
  >>> n = vector_norm(v)
  >>> numpy.allclose(n, numpy.linalg.norm(v))
  True
  >>> v = numpy.random.rand(6, 5, 3)
  >>> n = vector_norm(v, axis=-1)
  >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
  True
  >>> n = vector_norm(v, axis=1)
  >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
  True
  >>> v = numpy.random.rand(5, 4, 3)
  >>> n = numpy.empty((5, 3))
  >>> vector_norm(v, axis=1, out=n)
  >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
  True
  >>> vector_norm([])
  0.0
  >>> vector_norm([1])
  1.0

  """
  data = np.array(data, dtype=np.float64, copy=True)
  if out is None:
    if data.ndim == 1:
      return np.sqrt(np.dot(data, data))
    data *= data
    out = np.atleast_1d(np.sum(data, axis=axis))
    return np.sqrt(out)
  else:
    assert(False)
    #data *= data
    #numpy.sum(data, axis=axis, out=out)
    #numpy.sqrt(out, out)

def unit_vector(data, axis=None, out=None):
  """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

  COMMENT(@cpgoodri): For now, I am only implementing this for axis=None and out=None. I don't *think* this will affect internal functionality.

  >>> v0 = numpy.random.random(3)
  >>> v1 = unit_vector(v0)
  >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
  True
  >>> v0 = numpy.random.rand(5, 4, 3)
  >>> v1 = unit_vector(v0, axis=-1)
  >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
  >>> numpy.allclose(v1, v2)
  True
  >>> v1 = unit_vector(v0, axis=1)
  >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
  >>> numpy.allclose(v1, v2)
  True
  >>> v1 = numpy.empty((5, 4, 3))
  >>> unit_vector(v0, axis=1, out=v1)
  >>> numpy.allclose(v1, v2)
  True
  >>> list(unit_vector([]))
  []
  >>> list(unit_vector([1]))
  [1.0]

  """
  if out is None:
    data = np.array(data, dtype=np.float64, copy=True)
    if data.ndim == 1:
      data /= np.sqrt(np.dot(data, data))
      return data
  else:
    assert(False)
    #if out is not data:
    #  out[:] = numpy.array(data, copy=False)
    #data = out
  length = np.atleast_1d(np.sum(data*data, axis))
  #np.sqrt(length, length)
  length = np.sqrt(length)
  if axis is not None:
    assert(False)
    #length = numpy.expand_dims(length, axis)
  data /= length
  if out is None:
    return data


def random_vector(size):
  """Return array of random doubles in the half-open interval [0.0, 1.0).
  #COMMENT(cpgoodri): not jax compatible... TODO
  #TODO(cpgoodri): tests

  >>> v = random_vector(10000)
  >>> numpy.all(v >= 0) and numpy.all(v < 1)
  True
  >>> v0 = random_vector(10)
  >>> v1 = random_vector(10)
  >>> numpy.any(v0 == v1)
  False

  """
  return np.array(onp.random.random(size))


def vector_product(v0, v1, axis=0):
    """Return vector perpendicular to vectors.

    >>> v = vector_product([2, 0, 0], [0, 3, 0])
    >>> numpy.allclose(v, [0, 0, 6])
    True
    >>> v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    >>> v1 = [[3], [0], [0]]
    >>> v = vector_product(v0, v1)
    >>> numpy.allclose(v, [[0, 0, 0, 0], [0, 0, 6, 6], [0, -6, 0, -6]])
    True
    >>> v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    >>> v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    >>> v = vector_product(v0, v1, axis=1)
    >>> numpy.allclose(v, [[0, 0, 6], [0, -6, 0], [6, 0, 0], [0, -6, 6]])
    True

    """
    v0 = np.array(v0)
    v1 = np.array(v1)
    return np.cross(v0, v1, axis=axis)


def angle_between_vectors(v0, v1, directed=True, axis=0):
  """Return angle between vectors.

  If directed is False, the input vectors are interpreted as undirected axes,
  i.e. the maximum angle is pi/2.

  >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3])
  >>> numpy.allclose(a, math.pi)
  True
  >>> a = angle_between_vectors([1, -2, 3], [-1, 2, -3], directed=False)
  >>> numpy.allclose(a, 0)
  True
  >>> v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
  >>> v1 = [[3], [0], [0]]
  >>> a = angle_between_vectors(v0, v1)
  >>> numpy.allclose(a, [0, 1.5708, 1.5708, 0.95532])
  True
  >>> v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
  >>> v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
  >>> a = angle_between_vectors(v0, v1, axis=1)
  >>> numpy.allclose(a, [1.5708, 1.5708, 1.5708, 0.95532])
  True

  """
  v0 = np.array(v0, dtype=np.float64, copy=False)
  v1 = np.array(v1, dtype=np.float64, copy=False)
  dot = np.sum(v0 * v1, axis=axis)
  dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
  dot = np.clip(dot, -1.0, 1.0)
  return np.arccos(dot if directed else np.fabs(dot))


def inverse_matrix(matrix):
  """Return inverse of square transformation matrix.

  >>> M0 = random_rotation_matrix()
  >>> M1 = inverse_matrix(M0.T)
  >>> numpy.allclose(M1, numpy.linalg.inv(M0.T))
  True
  >>> for size in range(1, 7):
  ...     M0 = numpy.random.rand(size, size)
  ...     M1 = inverse_matrix(M0)
  ...     if not numpy.allclose(M1, numpy.linalg.inv(M0)): print(size)

  """
  return np.linalg.inv(matrix)


def concatenate_matrices(*matrices):
  """Return concatenation of series of transformation matrices.

  >>> M = numpy.random.rand(16).reshape((4, 4)) - 0.5
  >>> numpy.allclose(M, concatenate_matrices(M))
  True
  >>> numpy.allclose(numpy.dot(M, M.T), concatenate_matrices(M, M.T))
  True

  """
  M = np.identity(4)
  for i in matrices:
      M = np.dot(M, i)
  return M


def is_same_transform(matrix0, matrix1):
  """Return True if two matrices perform same transformation.

  >>> is_same_transform(numpy.identity(4), numpy.identity(4))
  True
  >>> is_same_transform(numpy.identity(4), random_rotation_matrix())
  False

  """
  #TODO(cpgoodri): tests
  matrix0 = np.array(matrix0, dtype=np.float64, copy=True)
  matrix0 /= matrix0[3, 3]
  matrix1 = np.array(matrix1, dtype=np.float64, copy=True)
  matrix1 /= matrix1[3, 3]
  return np.allclose(matrix0, matrix1)


def is_same_quaternion(q0, q1):
  """Return True if two quaternions are equal."""
  #TODO(cpgoodri): tests
  q0 = np.array(q0)
  q1 = np.array(q1)
  return np.allclose(q0, q1) or np.allclose(q0, -q1)


