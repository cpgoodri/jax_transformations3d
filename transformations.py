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








