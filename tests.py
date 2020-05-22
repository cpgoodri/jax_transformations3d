from transformations import *

from absl.testing import absltest
from absl.testing import parameterized

from jax import test_util as jtu

from jax.config import config as jax_config
jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS




class TransformationsTest(jtu.JaxTestCase):
  def test_identity_matrix(self):
    I = identity_matrix()
    self.assertAllClose(I, np.dot(I, I), True)
    self.assertAllClose(np.array([np.sum(I), np.trace(I)]), np.array([4.0, 4.0]), True)
    self.assertAllClose(I, np.identity(4), True)


  def test_translation_matrix(self):
    v = np.array(onp.random.random(3)) - 0.5
    self.assertAllClose(v, translation_matrix(v)[:3, 3], True)


  def test_translation_from_matrix(self):
    v0 = np.array(onp.random.random(3)) - 0.5
    v1 = translation_from_matrix(translation_matrix(v0))
    self.assertAllClose(v0, v1, True)
    

  def test_reflection_matrix(self):
    v0 = onp.random.random(4) - 0.5
    v0[3] = 1.
    v0 = np.array(v0)
    v1 = np.array(onp.random.random(3)) - 0.5
    R = reflection_matrix(v0, v1)
    self.assertAllClose(2, np.trace(R), False)
    self.assertAllClose(v0, np.dot(R, v0), True)
    v2 = onp.array(v0).copy()
    v3 = onp.array(v0).copy()
    v2[:3] += v1
    v3[:3] -= v1
    v2 = np.array(v2)
    v3 = np.array(v3)
    self.assertAllClose(v2, np.dot(R, v3), True)


  def test_reflection_from_matrix(self):
    v0 = np.array(onp.random.random(3)) - 0.5
    v1 = np.array(onp.random.random(3)) - 0.5
    M0 = reflection_matrix(v0, v1)
    point, normal = reflection_from_matrix(M0)
    M1 = reflection_matrix(point, normal)
    #is_same_transform(M0, M1)
    M0 /= M0[3,3]
    M1 /= M1[3,3]
    self.assertAllClose(M0,M1,True)
    
  def test_rotation_matrix(self):
    R = rotation_matrix(math.pi/2, np.array([0, 0, 1]), np.array([1, 0, 0]))
    self.assertAllClose(np.dot(R, np.array([0, 0, 0, 1])), np.array([1, -1, 0, 1]), False)
    
    """
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












































  def test_unit_vector(self):
    v0 = np.array(onp.random.random(3))
    v1 = unit_vector(v0)
    self.assertAllClose(v1, v0 / np.linalg.norm(v0), True)
    #>>> v0 = numpy.random.rand(5, 4, 3)
    #>>> v1 = unit_vector(v0, axis=-1)
    #>>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    #>>> numpy.allclose(v1, v2)
    #True
    #>>> v1 = unit_vector(v0, axis=1)
    #>>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    #>>> numpy.allclose(v1, v2)
    #True
    #>>> v1 = numpy.empty((5, 4, 3))
    #>>> unit_vector(v0, axis=1, out=v1)
    ##>>> numpy.allclose(v1, v2)
    #True
    #>>> list(unit_vector([]))
    #[]
    self.assertAllClose(unit_vector([1]), np.array([1.0]), True)
    #[1.0]













if __name__ == '__main__':
  absltest.main()

