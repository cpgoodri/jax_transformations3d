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
  #M[:3, 3] = direction[:3]
  return M


def translation_from_matrix(matrix):
    """Return translation vector from translation matrix.

    >>> v0 = numpy.random.random(3) - 0.5
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> numpy.allclose(v0, v1)
    True

    """
    return np.array(matrix, copy=False)[:3, 3].copy()









