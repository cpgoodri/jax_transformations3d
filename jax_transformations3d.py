import numpy as onp
import jax.numpy as jnp
from jax import lax, vmap, random
import math


def _raiseValueError(msg):
  raise ValueError(msg)


def identity_matrix():
  """Return 4x4 identity/unit matrix.

  """
  return jnp.identity(4)


def translation_matrix(direction):
  """Return matrix to translate by direction vector.

  """
  M = jnp.identity(4)
  #M = index_update(M, index[:3,3], direction[:3])
  M = M.at[:3, 3].set(direction[:3])
  return M


def translation_from_matrix(matrix):
  """Return translation vector from translation matrix.

  """
  #return jnp.array(matrix, copy=False)[:3, 3].copy()
  return jnp.array(matrix, copy=True)[:3, 3]


def reflection_matrix(point, normal):
  """Return matrix to mirror at plane defined by point and normal vector.

  """
  normal = unit_vector(normal[:3])
  M = jnp.identity(4)
  M = M.at[:3, :3].add(-2.0 * jnp.outer(normal, normal))
  #M = index_add(M, index[:3,:3], -2.0*jnp.outer(normal,normal))
  M = M.at[:3, 3].set((2.0 * jnp.dot(point[:3], normal)) * normal)
  #M = index_update(M, index[:3,3], (2.0 * jnp.dot(point[:3], normal)) * normal)
  #M[:3, :3] -= 2.0 * numpy.outer(normal, normal)
  #M[:3, 3] = (2.0 * numpy.dot(point[:3], normal)) * normal
  return M


def reflection_from_matrix(matrix):
  """Return mirror plane point and normal vector from reflection matrix.

  """
  M = jnp.array(matrix, dtype=jnp.float64, copy=False)
  # normal: unit eigenvector corresponding to eigenvalue -1
  w, V = jnp.linalg.eig(M[:3, :3])
  i = jnp.where(abs(jnp.real(w) + 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no unit eigenvector corresponding to eigenvalue -1')
  normal = jnp.real(V[:, i[0]]).squeeze()
  # point: any unit eigenvector corresponding to eigenvalue 1
  w, V = jnp.linalg.eig(M)
  i = jnp.where(abs(jnp.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
  point = jnp.real(V[:, i[-1]]).squeeze()
  point /= point[3]
  return point, normal


def rotation_matrix(angle, direction, point=None):
  """Return matrix to rotate about axis defined by point and direction.

  """
  sina = jnp.sin(angle)
  cosa = jnp.cos(angle)
  direction = unit_vector(direction[:3])
  # rotation matrix around unit vector
  R = jnp.diag(jnp.array([cosa, cosa, cosa]))
  R = R + jnp.outer(direction, direction) * (1.0 - cosa)
  direction = direction * sina
  R = R + jnp.array([[0.0, -direction[2], direction[1]],
                     [direction[2], 0.0, -direction[0]],
                     [-direction[1], direction[0], 0.0]])
  M = jnp.identity(4)
  M = M.at[:3, :3].set(R)
  #M = index_update(M, index[:3, :3], R)
  if point is not None:
    # rotation not around origin
    point = jnp.array(point[:3], dtype=jnp.float64, copy=False)
    M = M.at[:3, 3].set(point - jnp.dot(R, point))
    #M = index_update(M, index[:3, 3], point - jnp.dot(R, point))
  return M


def rotation_from_matrix(matrix):
  """Return rotation angle and axis from rotation matrix.

  """
  R = jnp.array(matrix, dtype=jnp.float64, copy=False)
  R33 = R[:3, :3]
  # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
  w, W = jnp.linalg.eig(jnp.transpose(R33))
  i = jnp.where(abs(jnp.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
  direction = jnp.real(W[:, i[-1]]).squeeze()
  # point: unit eigenvector of R33 corresponding to eigenvalue of 1
  w, Q = jnp.linalg.eig(R)
  i = jnp.where(abs(jnp.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no unit eigenvector corresponding to eigenvalue 1')
  point = jnp.real(Q[:, i[-1]]).squeeze()
  point /= point[3]
  # rotation angle depending on direction
  cosa = (jnp.trace(R33) - 1.0) / 2.0
  if abs(direction[2]) > 1e-8:
    sina = (R[1, 0] + (cosa - 1.0) * direction[0] * direction[1]) / direction[2]
  elif abs(direction[1]) > 1e-8:
    sina = (R[0, 2] + (cosa - 1.0) * direction[0] * direction[2]) / direction[1]
  else:
    sina = (R[2, 1] + (cosa - 1.0) * direction[1] * direction[2]) / direction[0]
  angle = jnp.arctan2(sina, cosa)
  return angle, direction, point


def scale_matrix(factor, origin=None, direction=None):
  """Return matrix to scale by factor around origin in direction.

  Use factor -1 for point symmetry.

  """
  if direction is None:
    # uniform scaling
    M = jnp.diag(jnp.array([factor, factor, factor, 1.0]))
    if origin is not None:
      M = M.at[:3, 3].set(origin[:3] * (1.0 - factor))
      #M = index_update(M, index[:3, 3], origin[:3])
      #M = index_update(M, index[:3, 3], M[:3, 3] * (1.0 - factor))
      #M[:3, 3] = origin[:3]
      #M[:3, 3] *= 1.0 - factor
  else:
    # nonuniform scaling
    direction = unit_vector(direction[:3])
    factor = 1.0 - factor
    M = jnp.identity(4)
    M = M.at[:3, :3].add(-factor * jnp.outer(direction, direction))
    #M = index_add(M, index[:3, :3], -factor * jnp.outer(direction, direction))
    #M[:3, :3] -= factor * jnp.outer(direction, direction)
    if origin is not None:
      M = M.at[:3, 3].set((factor * jnp.dot(origin[:3], direction)) * direction)
      #M = index_update(M, index[:3, 3], (factor * jnp.dot(origin[:3], direction)) * direction)
      #M[:3, 3] = (factor * numpy.dot(origin[:3], direction)) * direction
  return M


def scale_from_matrix(matrix):
  """Return scaling factor, origin and direction from scaling matrix.

  """
  M = jnp.array(matrix, dtype=jnp.float64, copy=False)
  M33 = M[:3, :3]
  factor = jnp.trace(M33) - 2.0
  try:
    # direction: unit eigenvector corresponding to eigenvalue factor
    w, V = jnp.linalg.eig(M33)
    i = jnp.where(abs(jnp.real(w) - factor) < 1e-8)[0][0]
    direction = jnp.real(V[:, i]).squeeze()
    direction /= vector_norm(direction)

  #WARNING(@cpgoodri): I'm not sure if this error-handling approach works with JAX, but it seems to pass tests...
  except IndexError:
    # uniform scaling
    factor = (factor + 2.0) / 3.0
    direction = None
  # origin: any eigenvector corresponding to eigenvalue 1
  w, V = jnp.linalg.eig(M)
  i = jnp.where(abs(jnp.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no eigenvector corresponding to eigenvalue 1')
  origin = jnp.real(V[:, i[-1]]).squeeze()
  origin /= origin[3]
  return factor, origin, direction


def projection_matrix(point,
                      normal,
                      direction=None,
                      perspective=None,
                      pseudo=False):
  """Return matrix to project onto plane defined by point and normal.

  Using either perspective point, projection direction, or none of both.

  If pseudo is True, perspective projections will preserve relative depth
  such that Perspective = dot(Orthogonal, PseudoPerspective).

  """
  M = jnp.identity(4)
  point = jnp.array(point[:3], dtype=jnp.float64, copy=False)
  normal = unit_vector(normal[:3])
  if perspective is not None:
    # perspective projection
    perspective = jnp.array(perspective[:3], dtype=jnp.float64, copy=False)
    temp = jnp.dot(perspective - point, normal)
    M = M.at[0, 0].set(temp)
    M = M.at[1, 1].set(temp)
    M = M.at[2, 2].set(temp)
    M = M.at[:3, :3].add(-jnp.outer(perspective, normal))
    #M = index_update(M, index[0, 0], temp)
    #M = index_update(M, index[1, 1], temp)
    #M = index_update(M, index[2, 2], temp)
    #M = index_add(M, index[:3, :3], -jnp.outer(perspective, normal))
    if pseudo:
      # preserve relative depth
      M = M.at[:3, :3].add(-jnp.outer(normal, normal))
      #M = index_add(M, index[:3, :3], -jnp.outer(normal, normal))
      M = M.at[:3, 3].set(jnp.dot(point, normal) * (perspective + normal))
      #M = index_update(M, index[:3, 3], jnp.dot(point, normal) * (perspective+normal))
    else:
      M = M.at[:3, 3].set(jnp.dot(point, normal) * perspective)
      #M = index_update(M, index[:3, 3], jnp.dot(point, normal) * perspective)
    M = M.at[3, :3].set(-normal)
    #M = index_update(M, index[3, :3], -normal)
    M = M.at[3, 3].set(jnp.dot(perspective, normal))
    #M = index_update(M, index[3, 3], jnp.dot(perspective, normal))
  elif direction is not None:
    # parallel projection
    direction = jnp.array(direction[:3], dtype=jnp.float64, copy=False)
    scale = jnp.dot(direction, normal)
    M = M.at[:3, :3].add(-jnp.outer(direction, normal) / scale)
    #M = index_add(M, index[:3, :3], -jnp.outer(direction, normal) / scale)
    M = M.at[:3, 3].set(direction * (jnp.dot(point, normal) / scale))
  else:
    # orthogonal projection
    M = M.at[:3, :3].add(-jnp.outer(normal, normal))
    #M = index_add(M, index[:3, :3], -jnp.outer(normal, normal))
    M = M.at[:3, 3].set(jnp.dot(point, normal) * normal)
    #M = index_update(M, index[:3, 3], jnp.dot(point, normal) * normal)
  return M


def projection_from_matrix(matrix, pseudo=False):
  """Return projection plane and perspective point from projection matrix.

  Return values are same as arguments for projection_matrix function:
  point, normal, direction, perspective, and pseudo.

  """
  M = jnp.array(matrix, dtype=jnp.float64, copy=False)
  M33 = M[:3, :3]
  w, V = jnp.linalg.eig(M)
  i = jnp.where(abs(jnp.real(w) - 1.0) < 1e-8)[0]
  if not pseudo and len(i):
    # point: any eigenvector corresponding to eigenvalue 1
    point = jnp.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    # direction: unit eigenvector corresponding to eigenvalue 0
    w, V = jnp.linalg.eig(M33)
    i = jnp.where(abs(jnp.real(w)) < 1e-8)[0]
    if not len(i):
      raise ValueError('no eigenvector corresponding to eigenvalue 0')
    direction = jnp.real(V[:, i[0]]).squeeze()
    direction /= vector_norm(direction)
    # normal: unit eigenvector of M33.T corresponding to eigenvalue 0
    w, V = jnp.linalg.eig(M33.T)
    i = jnp.where(abs(jnp.real(w)) < 1e-8)[0]
    if len(i):
      # parallel projection
      normal = jnp.real(V[:, i[0]]).squeeze()
      normal /= vector_norm(normal)
      return point, normal, direction, None, False
    else:
      # orthogonal projection, where normal equals direction vector
      return point, direction, None, None, False
  else:
    # perspective projection
    i = jnp.where(abs(jnp.real(w)) > 1e-8)[0]
    if not len(i):
      raise ValueError('no eigenvector not corresponding to eigenvalue 0')
    point = jnp.real(V[:, i[-1]]).squeeze()
    point /= point[3]
    normal = -M[3, :3]
    perspective = M[:3, 3] / jnp.dot(point[:3], normal)
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

  """
  normal = unit_vector(normal[:3])
  direction = unit_vector(direction[:3])

  #lax.cond(jnp.abs(jnp.dot(normal, direction)) > 1e-6,
  #    'direction and normal vectors are not orthogonal', _raiseValueError,
  #    None, lambda x: None)
  if jnp.abs(jnp.dot(normal, direction)) > 1e-6:
    raise ValueError('direction and normal vectors are not orthogonal')
  angle = jnp.tan(angle)
  M = jnp.identity(4)
  M = M.at[:3, :3].add(angle * jnp.outer(direction, normal))
  #M = index_add(M, index[:3, :3], angle * jnp.outer(direction, normal))
  #M[:3, :3] += angle * numpy.outer(direction, normal)
  M = M.at[:3, 3].set(-angle * jnp.dot(point[:3], normal) * direction)
  #M = index_update(M, index[:3, 3], -angle * jnp.dot(point[:3], normal) * direction)
  #M[:3, 3] = -angle * numpy.dot(point[:3], normal) * direction
  return M


def shear_from_matrix(matrix):
  """Return shear angle, direction and plane from shear matrix.

  """
  M = jnp.array(matrix, dtype=jnp.float64, copy=False)
  M33 = M[:3, :3]
  # normal: cross independent eigenvectors corresponding to the eigenvalue 1
  w, V = jnp.linalg.eig(M33)
  i = jnp.where(abs(jnp.real(w) - 1.0) < 1e-4)[0]
  if len(i) < 2:
    raise ValueError('no two linear independent eigenvectors found %s' % w)
  V = jnp.real(V[:, i]).squeeze().T
  lenorm = -1.0
  for i0, i1 in ((0, 1), (0, 2), (1, 2)):
    n = jnp.cross(V[i0], V[i1])
    w = vector_norm(n)
    if w > lenorm:
      lenorm = w
      normal = n
  normal /= lenorm
  # direction and angle
  direction = jnp.dot(M33 - jnp.identity(3), normal)
  angle = vector_norm(direction)
  direction /= angle
  angle = jnp.arctan(angle)
  # point: eigenvector corresponding to eigenvalue 1
  w, V = jnp.linalg.eig(M)
  i = jnp.where(abs(jnp.real(w) - 1.0) < 1e-8)[0]
  if not len(i):
    raise ValueError('no eigenvector corresponding to eigenvalue 1')
  point = jnp.real(V[:, i[-1]]).squeeze()
  point /= point[3]
  return angle, direction, point, normal


def decompose_matrix(matrix):
  print("WARNING: not implemented.")
  raise NotImplementedError


def compose_matrix(scale=None,
                   shear=None,
                   angles=None,
                   translate=None,
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

  """
  try:
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
  except (AttributeError, KeyError):
    _TUPLE2AXES[axes]  # noqa: validation
    firstaxis, parity, repetition, frame = axes

  i = firstaxis
  j = _NEXT_AXIS[i + parity]
  k = _NEXT_AXIS[i - parity + 1]

  if frame:
    ai, ak = ak, ai
  if parity:
    ai, aj, ak = -ai, -aj, -ak

  si, sj, sk = jnp.sin(ai), jnp.sin(aj), jnp.sin(ak)
  ci, cj, ck = jnp.cos(ai), jnp.cos(aj), jnp.cos(ak)
  cc, cs = ci * ck, ci * sk
  sc, ss = si * ck, si * sk

  M = jnp.identity(4)
  if repetition:
    M = M.at[i, i].set(cj)
    #M[i, i] = cj
    M = M.at[i, j].set(sj * si)
    #M[i, j] = sj*si
    M = M.at[i, k].set(sj * ci)
    #M[i, k] = sj*ci
    M = M.at[j, i].set(sj * sk)
    #M[j, i] = sj*sk
    M = M.at[j, j].set(-cj * ss + cc)
    #M[j, j] = -cj*ss+cc
    M = M.at[j, k].set(-cj * cs - sc)
    #M[j, k] = -cj*cs-sc
    M = M.at[k, i].set(-sj * ck)
    #M[k, i] = -sj*ck
    M = M.at[k, j].set(cj * sc + cs)
    #M[k, j] = cj*sc+cs
    M = M.at[k, k].set(cj * cc - ss)
    #M[k, k] = cj*cc-ss
  else:
    M = M.at[i, i].set(cj * ck)
    #M[i, i] = cj*ck
    M = M.at[i, j].set(sj * sc - cs)
    #M[i, j] = sj*sc-cs
    M = M.at[i, k].set(sj * cc + ss)
    #M[i, k] = sj*cc+ss
    M = M.at[j, i].set(cj * sk)
    #M[j, i] = cj*sk
    M = M.at[j, j].set(sj * ss + cc)
    #M[j, j] = sj*ss+cc
    M = M.at[j, k].set(sj * cs - sc)
    #M[j, k] = sj*cs-sc
    M = M.at[k, i].set(-sj)
    #M[k, i] = -sj
    M = M.at[k, j].set(cj * si)
    #M[k, j] = cj*si
    M = M.at[k, k].set(cj * ci)
    #M[k, k] = cj*ci
  return M


def euler_from_matrix(matrix, axes='sxyz'):
  """Return Euler angles from rotation matrix for specified axis sequence.

  axes : One of 24 axis sequences as string or encoded tuple

  Note that many Euler angle triplets can describe one matrix.

  """
  try:
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
  except (AttributeError, KeyError):
    _TUPLE2AXES[axes]  # noqa: validation
    firstaxis, parity, repetition, frame = axes

  i = firstaxis
  j = _NEXT_AXIS[i + parity]
  k = _NEXT_AXIS[i - parity + 1]

  M = jnp.array(matrix, dtype=jnp.float64, copy=False)[:3, :3]
  if repetition:
    sy = jnp.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
    ax, ay, az = lax.cond(
        sy > _EPS, (jnp.arctan2(M[i, j], M[i, k]), jnp.arctan2(
            sy, M[i, i]), jnp.arctan2(M[j, i], -M[k, i])), lambda x: x,
        (jnp.arctan2(-M[j, k], M[j, j]), jnp.arctan2(sy, M[i, i]), 0.0),
        lambda x: x)
    """
    if sy > _EPS:
      ax = jnp.arctan2( M[i, j],  M[i, k])
      ay = jnp.arctan2( sy,       M[i, i])
      az = jnp.arctan2( M[j, i], -M[k, i])
    else:
      ax = jnp.arctan2(-M[j, k],  M[j, j])
      ay = jnp.arctan2( sy,       M[i, i])
      az = 0.0
    """
  else:
    cy = jnp.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
    ax, ay, az = lax.cond(
        cy > _EPS, (jnp.arctan2(M[k, j], M[k, k]), jnp.arctan2(
            -M[k, i], cy), jnp.arctan2(M[j, i], M[i, i])), lambda x: x,
        (jnp.arctan2(-M[j, k], M[j, j]), jnp.arctan2(-M[k, i], cy), 0.0),
        lambda x: x)
    """
    if cy > _EPS:
      ax = jnp.arctan2( M[k, j],  M[k, k])
      ay = jnp.arctan2(-M[k, i],  cy)
      az = jnp.arctan2( M[j, i],  M[i, i])
    else:
      ax = jnp.arctan2(-M[j, k],  M[j, j])
      ay = jnp.arctan2(-M[k, i],  cy)
      az = 0.0
    """

  if parity:
    ax, ay, az = -ax, -ay, -az
  if frame:
    ax, az = az, ax
  return ax, ay, az


def euler_from_quaternion(quaternion, axes='sxyz'):
  """Return Euler angles from quaternion for specified axis sequence.

  """
  return euler_from_matrix(quaternion_matrix(quaternion), axes)


def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
  """Return quaternion from Euler angles and axis sequence.

  ai, aj, ak : Euler's roll, pitch and yaw angles
  axes : One of 24 axis sequences as string or encoded tuple

  """
  try:
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
  except (AttributeError, KeyError):
    _TUPLE2AXES[axes]  # noqa: validation
    firstaxis, parity, repetition, frame = axes

  i = firstaxis + 1
  j = _NEXT_AXIS[i + parity - 1] + 1
  k = _NEXT_AXIS[i - parity] + 1

  if frame:
    ai, ak = ak, ai
  if parity:
    aj = -aj

  ai /= 2.0
  aj /= 2.0
  ak /= 2.0
  ci = jnp.cos(ai)
  si = jnp.sin(ai)
  cj = jnp.cos(aj)
  sj = jnp.sin(aj)
  ck = jnp.cos(ak)
  sk = jnp.sin(ak)
  cc = ci * ck
  cs = ci * sk
  sc = si * ck
  ss = si * sk

  q = jnp.empty((4,))
  if repetition:
    q = q.at[0].set(cj * (cc - ss))
    #q[0] = cj*(cc - ss)
    q = q.at[i].set(cj * (cs + sc))
    #q[i] = cj*(cs + sc)
    q = q.at[j].set(sj * (cc + ss))
    #q[j] = sj*(cc + ss)
    q = q.at[k].set(sj * (cs - sc))
    #q[k] = sj*(cs - sc)
  else:
    q = q.at[0].set(cj * cc + sj * ss)
    #q[0] = cj*cc + sj*ss
    q = q.at[i].set(cj * sc - sj * cs)
    #q[i] = cj*sc - sj*cs
    q = q.at[j].set(cj * ss + sj * cc)
    #q[j] = cj*ss + sj*cc
    q = q.at[k].set(cj * cs - sj * sc)
    #q[k] = cj*cs - sj*sc
  if parity:
    q = q.at[j].multiply(-1.0)
    #q = index_update(q, index[j], -1.0 * q[j])
    #q[j] *= -1.0

  return q


def quaternion_about_axis(angle, axis):
  """Return quaternion for rotation about axis.

  """
  q = jnp.array([0.0, axis[0], axis[1], axis[2]])
  qlen = vector_norm(q)
  q = lax.cond(qlen > _EPS,
               q * jnp.sin(angle / 2.0) / qlen, lambda x: x, q, lambda x: x)
  #if qlen > _EPS:
  #  q *= math.sin(angle/2.0) / qlen
  q = q.at[0].set(jnp.cos(angle / 2.0))
  #q = index_update(q, index[0], jnp.cos(angle/2.0))
  #q[0] = math.cos(angle/2.0)
  return q


def quaternion_matrix(quaternion):
  """Return homogeneous rotation matrix from quaternion.

  """
  q = jnp.array(quaternion, dtype=jnp.float64, copy=True)
  n = jnp.dot(q, q)

  def calc_mat_posn(qn):
    q, n = qn
    q *= jnp.sqrt(2.0 / n)
    q = jnp.outer(q, q)
    return jnp.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
         [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
         [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
         [0.0, 0.0, 0.0, 1.0]])

  return lax.cond(n < _EPS, jnp.identity(4), lambda x: x, (q, n), calc_mat_posn)
  """
  if n < _EPS:
      return jnp.identity(4)
  q *= jnp.sqrt(2.0 / n)
  q = jnp.outer(q, q)
  return jnp.array([
      [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
      [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
      [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
      [                0.0,                 0.0,                 0.0, 1.0]])
  """


def quaternion_from_matrix(matrix, isprecise=False):
  """Return quaternion from rotation matrix.

  If isprecise is True, the input matrix is assumed to be a precise rotation
  matrix and a faster algorithm is used.

  """
  M = jnp.array(matrix, dtype=jnp.float64, copy=False)[:4, :4]
  if isprecise:
    q = jnp.empty((4,))
    t = jnp.trace(M)

    def case1(Mt):
      M, t = Mt
      return jnp.array(
          [t, M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]]), t

    def case2(Mtq):
      M, t, qtemp = Mtq
      i, j, k = lax.cond(M[1, 1] > M[0, 0], (1, 2, 0), lambda x: x, (0, 1, 2),
                         lambda x: x)
      i, j, k = lax.cond(M[2, 2] > M[i, i], (2, 0, 1), lambda x: x, (i, j, k),
                         lambda x: x)
      t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
      qtemp = qtemp.at[i].set(t)
      qtemp = qtemp.at[j].set(M[i, j] + M[j, i])
      qtemp = qtemp.at[k].set(M[k, i] + M[i, k])
      qtemp = qtemp.at[3].set(M[k, j] - M[j, k])
      qtemp = qtemp[jnp.array([3, 0, 1, 2])]
      return qtemp, t

    q, t = lax.cond(t > M[3, 3], (M, t), case1, (M, t, q), case2)
    """
      if t > M[3, 3]:
        q = jnp.array([t,
          M[2, 1] - M[1, 2],
          M[0, 2] - M[2, 0],
          M[1, 0] - M[0, 1]])

        #q[0] = t
        #q[3] = M[1, 0] - M[0, 1]
        #q[2] = M[0, 2] - M[2, 0]
        #q[1] = M[2, 1] - M[1, 2]
      else:
        i, j, k = lax.cond( M[1,1] > M[0,0],
            (1,2,0), lambda x: x,
            (0,1,2), lambda x: x)
        i, j, k = lax.cond( M[2,2] > M[i,i],
            (2,0,1), lambda x: x,
            (i,j,k), lambda x: x)
        #i, j, k = 0, 1, 2
        #if M[1, 1] > M[0, 0]:
        #  i, j, k = 1, 2, 0
        #if M[2, 2] > M[i, i]:
        #  i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q = index_update(q, index[i], t)
        #q[i] = t
        q = index_update(q, index[j], M[i, j] + M[j, i])
        #q[j] = M[i, j] + M[j, i]
        q = index_update(q, index[k], M[k, i] + M[i, k])
        #q[k] = M[k, i] + M[i, k]
        q = index_update(q, index[3], M[k, j] - M[j, k])
        #q[3] = M[k, j] - M[j, k]
        q = q[[3, 0, 1, 2]]
      """
    q = q * (0.5 / jnp.sqrt(t * M[3, 3]))
  else:
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]
    # symmetric matrix K
    K = jnp.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                   [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                   [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                   [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0
    # quaternion is eigenvector of K that corresponds to largest eigenvalue
    w, V = jnp.linalg.eigh(K, UPLO='L', symmetrize_input=False)
    q = V[[3, 0, 1, 2], jnp.argmax(w)]
  #q = q * jnp.sign(q[0])
  q = lax.cond(q[0] < 0.0, -1.0 * q, lambda x: x, q, lambda x: x)
  #if q[0] < 0.0:
  #  q = jnp.negative(q)
  return q


def quaternion_multiply(quaternion1, quaternion0):
  """Return multiplication of two quaternions.

  """
  w0, x0, y0, z0 = quaternion0
  w1, x1, y1, z1 = quaternion1
  return jnp.array([
      -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0, x1 * w0 + y1 * z0 - z1 * y0 +
      w1 * x0, -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
      x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
  ],
                   dtype=jnp.float64)


def quaternion_conjugate(quaternion):
  """Return conjugate of quaternion.

  """
  #q = jnp.array(quaternion, dtype=jnp.float64, copy=True)
  q = quaternion.at[1:].set(-quaternion[1:])
  #q = index_update(quaternion, index[1:], -quaternion[1:])
  #numpy.negative(q[1:], q[1:])
  return q


def quaternion_inverse(quaternion):
  """Return inverse of quaternion.

  """
  #q = jnp.array(quaternion, dtype=numpy.float64, copy=True)
  #numpy.negative(q[1:], q[1:])
  q = quaternion_conjugate(quaternion)
  return q / jnp.dot(q, q)


def quaternion_real(quaternion):
  """Return real part of quaternion.

  """
  return quaternion[0]
  #return jnp.float64(quaternion[0])


def quaternion_imag(quaternion):
  """Return imaginary part of quaternion.

  """
  return jnp.array(quaternion[1:4], dtype=jnp.float64, copy=True)


def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
  print("WARNING: not implemented.")
  raise NotImplementedError


#############################################
################### Helpers #################
#############################################


def random_quaternion(rand=None, key=None):
  """Return uniform random unit quaternion.

  rand: array like or None
      Three independent random variables that are uniformly distributed
      between 0 and 1.
  key:  key for jax.random or None

  If rand is not None, use those values to create a quaternion
  If rand is None and key is not None, use jax.random to
    create a quaternion
  If both rand and key are None, fall back to onp.random

  """
  if (rand is None and key is None):
    rnd = jnp.array(onp.random.rand(3), dtype=jnp.float64)

  elif (rand is None):
    rnd = random.uniform(key, (3,), minval=0.0, maxval=1.0, dtype=jnp.float64)

  else:
    rnd = jnp.array(rand, dtype=jnp.float64)
    assert (rnd.shape == (3,))
    #assert len(rnd) == 3

  r1 = jnp.sqrt(1.0 - rnd[0])
  r2 = jnp.sqrt(rnd[0])
  pi2 = jnp.pi * 2.0
  t1 = pi2 * rnd[1]
  t2 = pi2 * rnd[2]
  return jnp.array(
      [jnp.cos(t2) * r2,
       jnp.sin(t1) * r1,
       jnp.cos(t1) * r1,
       jnp.sin(t2) * r2])


def random_rotation_matrix(rand=None, key=None):
  """Return uniform random rotation matrix.

  rand: array like
      Three independent random variables that are uniformly distributed
      between 0 and 1 for each returned quaternion.

  """
  return quaternion_matrix(random_quaternion(rand, key))


# epsilon for testing whether a number is close to zero
_EPS = jnp.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0),
    'sxyx': (0, 0, 1, 0),
    'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0),
    'syzx': (1, 0, 0, 0),
    'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0),
    'syxy': (1, 1, 1, 0),
    'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0),
    'szyx': (2, 1, 0, 0),
    'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1),
    'rxyx': (0, 0, 1, 1),
    'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1),
    'rxzy': (1, 0, 0, 1),
    'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1),
    'ryxy': (1, 1, 1, 1),
    'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1),
    'rxyz': (2, 1, 0, 1),
    'rzyz': (2, 1, 1, 1)
}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def vector_norm(data, axis=None, out=None):
  """Return length, i.e. Euclidean norm, of ndarray along axis.

  COMMENT(@cpgoodri): For now, I am only implementing this for out=None. I don't *think* this will affect internal functionality.

  """
  data = jnp.array(data, dtype=jnp.float64, copy=True)
  if out is None:
    if data.ndim == 1:
      return jnp.sqrt(jnp.dot(data, data))
    data *= data
    out = jnp.atleast_1d(jnp.sum(data, axis=axis))
    return jnp.sqrt(out)
  else:
    assert (False)
    #data *= data
    #numpy.sum(data, axis=axis, out=out)
    #numpy.sqrt(out, out)


def unit_vector(data, axis=None, out=None):
  """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

  COMMENT(@cpgoodri): For now, I am only implementing this for axis=None and out=None. I don't *think* this will affect internal functionality.

  """
  if out is None:
    data = jnp.array(data, dtype=jnp.float64, copy=True)
    if data.ndim == 1:
      data /= jnp.sqrt(jnp.dot(data, data))
      return data
  else:
    assert (False)
    #if out is not data:
    #  out[:] = numpy.array(data, copy=False)
    #data = out
  length = jnp.atleast_1d(jnp.sum(data * data, axis))
  #jnp.sqrt(length, length)
  length = jnp.sqrt(length)
  if axis is not None:
    length = jnp.expand_dims(length, axis)
  data /= length
  return data


def random_vector(shape, key=None):
  """Return array of random doubles in the half-open interval [0.0, 1.0).
  Jax-compatible only if key is not None
  shape can be an int or a tuple
  This cannot be jitted, not 100% sure why

  """
  if (not isinstance(shape, tuple)):
    shape = (shape,)

  if (key is None):
    return jnp.array(onp.random.random(shape), dtype=jnp.float64)
  else:
    return random.uniform(key, shape, minval=0.0, maxval=1.0, dtype=jnp.float64)


def vector_product(v0, v1, axis=0):
  """Return vector perpendicular to vectors.

    """
  v0 = jnp.array(v0)
  v1 = jnp.array(v1)
  return jnp.cross(v0, v1, axis=axis)


def angle_between_vectors(v0, v1, directed=True, axis=0):
  """Return angle between vectors.

  If directed is False, the input vectors are interpreted as undirected axes,
  i.e. the maximum angle is pi/2.

  """
  v0 = jnp.array(v0, dtype=jnp.float64, copy=False)
  v1 = jnp.array(v1, dtype=jnp.float64, copy=False)
  dot = jnp.sum(v0 * v1, axis=axis)
  dot /= vector_norm(v0, axis=axis) * vector_norm(v1, axis=axis)
  dot = jnp.clip(dot, -1.0, 1.0)
  return jnp.arccos(dot if directed else jnp.fabs(dot))


def inverse_matrix(matrix):
  """Return inverse of square transformation matrix.

  """
  return jnp.linalg.inv(matrix)


def concatenate_matrices(*matrices):
  """Return concatenation of series of transformation matrices.

  """
  M = jnp.identity(4)
  for i in matrices:
    M = jnp.dot(M, i)
  return M


def is_same_transform(matrix0, matrix1):
  """Return True if two matrices perform same transformation.

  """
  #TODO(cpgoodri): tests
  matrix0 = jnp.array(matrix0, dtype=jnp.float64, copy=True)
  matrix0 /= matrix0[3, 3]
  matrix1 = jnp.array(matrix1, dtype=jnp.float64, copy=True)
  matrix1 /= matrix1[3, 3]
  return jnp.allclose(matrix0, matrix1)


def is_same_quaternion(q0, q1):
  """Return True if two quaternions are equal."""
  #TODO(cpgoodri): tests
  q0 = jnp.array(q0)
  q1 = jnp.array(q1)
  return jnp.allclose(q0, q1) or jnp.allclose(q0, -q1)


#############################################
############ New functionality ##############
#############################################


def matrix_apply(M, v):
  """ Multiply a vector or array of vectors by a transformation matrix

  Args:
    M: Transformation matrix with shape (4,4)
    v: Vector or array of vectors to be transformed. A single vector can have
      shape (3,) or (4,), and an array of m vectors can have shape (m,3) or
      (m,4). When v has shape (4,) or (m,4), it is not checked that the last
      element of each vector is 1.
  Return:
    Transformed vector or array of transformed vectors, with shape equal to
      v.shape.
  """

  assert (M.shape == (4, 4))

  if (v.shape[-1] == 4):
    return jnp.matmul(M, v.T).T
  elif (v.shape[-1] == 3):
    vtemp = jnp.pad(
        jnp.atleast_2d(v), ((0, 0), (0, 1)),
        mode='constant',
        constant_values=1.0)
    return jnp.reshape((jnp.matmul(M, vtemp.T)[:3]).T, v.shape)
  else:
    raise AssertionError('v must have shape (m,3) or (m,4) or (3,) or (4,)')


def quaternion_apply(q, v, index_start=0):
  """ Rotate a vector or array of vectors by a quaternion.

  Args:
    q: Rotation quaternion(s) with shape (4,) or (m,4). When shape is (m,4),
      each vector is rotated by a different quaternion
    v: Vector or array of m vectors to be rotated. Must have shape (l,) or
      (m,l) where l>=3. When l>3, the vector(s) to be rotated are
      v[index_start:index_start+3] or v[:,index_start:index_start+3]. All
      elements outside this range are ignored and copied into the returned
      array.

  Return:
    Rotated vector or array of rotated vectors. The precise meaning depends
    on the shapes of the input arrays:

    q.shape==(4,) and v.shape==(l,) --> output.shape==(l,)
      rotate v by q
    q.shape==(m,4) and v.shape==(l,) --> output.shape==(m,l)
      rotate v by each q
    q.shape==(4,) and v.shape==(m,l) --> output.shape==(m,l)
      rotate each v by q
    q.shape==(m,4) and v.shape==(m,l) --> output.shape==(m,l)
      rotate the ith v by the ith q

  """

  if (not (len(q.shape) == 1 or len(q.shape) == 2)):
    raise ValueError('q must either have 1 or 2 dimensions')
  if (q.shape[-1] != 4):
    raise ValueError('last dimension of q must have length 4')
  if (not (len(v.shape) == 1 or len(v.shape) == 2)):
    raise ValueError('v must either have 1 or 2 dimensions')
  if (v.shape[-1] < 3):
    raise ValueError(
        'last dimension of v must have length greater than or equal to 3')
  if (len(v.shape) == 2 and len(q.shape) == 2):
    if (v.shape[0] != q.shape[0]):
      raise ValueError('incompatible number of quaternions and vectors')

  def qapply(quat, vec):
    vec_as_quat = jnp.pad(
        vec[index_start:index_start + 3], (1, 0),
        mode='constant',
        constant_values=0.)
    #COMMENT(cpgoodri): it might be more efficient to do this by hand
    vec_rot = quaternion_multiply(
        quat, quaternion_multiply(vec_as_quat, quaternion_conjugate(quat)))[1:]
    return vec.at[index_start:index_start + 3].set(vec_rot)
    #return index_update(vec, index[index_start:index_start+3], vec_rot)

  if (len(q.shape) == 1 and len(v.shape) == 1):
    apply_fn = qapply
  elif (len(q.shape) == 2 and len(v.shape) == 1):
    apply_fn = vmap(qapply, in_axes=(0, None))
  elif (len(q.shape) == 1 and len(v.shape) == 2):
    apply_fn = vmap(qapply, in_axes=(None, 0))
  elif (len(q.shape) == 2 and len(v.shape) == 2):
    apply_fn = vmap(qapply, in_axes=(0, 0))
  else:
    raise AssertionError('Should be impossible to get here.')

  return apply_fn(q, v)

  #if(len(q.shape) == 1):
  #  apply_fn = vmap(qapply,in_axes=(None,0))
  #elif(len(q.shape) == 2):
  #  apply_fn = vmap(qapply,in_axes=(0,0))
  #else:
  #  raise ValueError('q must have shape (4,) or (m,4)')
  #return jnp.reshape(apply_fn(q, jnp.atleast_2d(v)),v.shape)
