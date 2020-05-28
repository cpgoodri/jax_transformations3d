import numpy as onp
import jax.numpy as np
from jax.ops import index, index_add, index_update
import math



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
  sina = math.sin(angle)
  cosa = math.cos(angle)
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
  angle = math.atan2(sina, cosa)
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










































def random_vector(size):
  """Return array of random doubles in the half-open interval [0.0, 1.0).
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
      return math.sqrt(np.dot(data, data))
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
      data /= math.sqrt(np.dot(data, data))
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






def is_same_transform(matrix0, matrix1):
  """Return True if two matrices perform same transformation.
  #TODO(cpgoodri): tests

  >>> is_same_transform(numpy.identity(4), numpy.identity(4))
  True
  >>> is_same_transform(numpy.identity(4), random_rotation_matrix())
  False

  """
  matrix0 = np.array(matrix0, dtype=np.float64, copy=True)
  matrix0 /= matrix0[3, 3]
  matrix1 = np.array(matrix1, dtype=np.float64, copy=True)
  matrix1 /= matrix1[3, 3]
  return np.allclose(matrix0, matrix1)



