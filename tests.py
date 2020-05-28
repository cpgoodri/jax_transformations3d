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
    v = random_vector(3) - 0.5
    self.assertAllClose(v, translation_matrix(v)[:3, 3], True)


  def test_translation_from_matrix(self):
    v0 = random_vector(3) - 0.5
    v1 = translation_from_matrix(translation_matrix(v0))
    self.assertAllClose(v0, v1, True)
    

  def test_reflection_matrix(self):
    v0 = onp.random.random(4) - 0.5
    v0[3] = 1.
    v0 = np.array(v0)
    v1 = random_vector(3) - 0.5
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
    v0 = random_vector(3) - 0.5
    v1 = random_vector(3) - 0.5
    M0 = reflection_matrix(v0, v1)
    point, normal = reflection_from_matrix(M0)
    M1 = reflection_matrix(point, normal)
    assert(is_same_transform(M0, M1))
   

  def test_rotation_matrix(self):
    R = rotation_matrix(math.pi/2, np.array([0, 0, 1]), np.array([1, 0, 0]))
    self.assertAllClose(np.dot(R, np.array([0, 0, 0, 1])), np.array([1, -1, 0, 1]), False)
    
    angle = (onp.random.random() - 0.5) * (2*math.pi)
    direc = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    R0 = rotation_matrix(angle, direc, point)
    R1 = rotation_matrix(angle-2*math.pi, direc, point)
    assert(is_same_transform(R0, R1))
    
    R0 = rotation_matrix(angle, direc, point)
    R1 = rotation_matrix(-angle, -direc, point)
    assert(is_same_transform(R0, R1))
    
    I = np.identity(4, np.float64)
    self.assertAllClose(I, rotation_matrix(math.pi*2, direc), True)
    
    self.assertAllClose(2, np.trace(rotation_matrix(math.pi/2, direc, point)), False)


  def test_rotation_from_matrix(self):
    angle = (onp.random.random() - 0.5) * (2*math.pi)
    direc = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    R0 = rotation_matrix(angle, direc, point)
    angle, direc, point = rotation_from_matrix(R0)
    R1 = rotation_matrix(angle, direc, point)
    assert(is_same_transform(R0, R1))


  def test_scale_matrix(self):
    v = (onp.random.rand(4, 5) - 0.5) * 20
    v[3] = 1
    v = np.array(v)
    S = scale_matrix(-1.234)
    self.assertAllClose(np.dot(S, v)[:3], -1.234*v[:3], True)
    
    factor = onp.random.random() * 10 - 5
    origin = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    S = scale_matrix(factor, origin)
    S = scale_matrix(factor, origin, direct)


  def test_scale_from_matrix(self):
    factor = onp.random.random() * 10 - 5
    origin = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    S0 = scale_matrix(factor, origin)
    factor, origin, direction = scale_from_matrix(S0)
    S1 = scale_matrix(factor, origin, direction)
    assert(is_same_transform(S0, S1))
    
    S0 = scale_matrix(factor, origin, direct)
    factor, origin, direction = scale_from_matrix(S0)
    S1 = scale_matrix(factor, origin, direction)
    assert(is_same_transform(S0, S1))


  def test_projection_matrix(self):
    P = projection_matrix([0, 0, 0], [1, 0, 0])
    self.assertAllClose(P[1:, 1:], np.identity(4)[1:, 1:], True)
    point  = random_vector(3) - 0.5
    normal = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    persp  = random_vector(3) - 0.5
    P0 = projection_matrix(point, normal)
    P1 = projection_matrix(point, normal, direction=direct)
    P2 = projection_matrix(point, normal, perspective=persp)
    P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    assert(is_same_transform(P2, np.dot(P0, P3)))
    P = projection_matrix([3, 0, 0], [1, 1, 0], [1, 0, 0])
    v0 = (onp.random.rand(4, 5) - 0.5) * 20
    v0[3] = 1
    v0 = np.array(v0)
    v1 = np.dot(P, v0)
    self.assertAllClose(v1[1], v0[1], True)
    self.assertAllClose(v1[0], 3-v1[1], True)


  def test_projection_from_matrix(self):
    point = random_vector(3) - 0.5
    normal = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    persp = random_vector(3) - 0.5
    P0 = projection_matrix(point, normal)
    result = projection_from_matrix(P0)
    P1 = projection_matrix(*result)
    assert(is_same_transform(P0, P1))
    P0 = projection_matrix(point, normal, direct)
    result = projection_from_matrix(P0)
    P1 = projection_matrix(*result)
    assert(is_same_transform(P0, P1))
    P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    result = projection_from_matrix(P0, pseudo=False)
    P1 = projection_matrix(*result)
    assert(is_same_transform(P0, P1))
    P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    result = projection_from_matrix(P0, pseudo=True)
    P1 = projection_matrix(*result)
    assert(is_same_transform(P0, P1))




































  def test_vector_norm(self):
    v = random_vector(3)
    n = vector_norm(v)
    self.assertAllClose(n, np.linalg.norm(v), True)
    
    v = np.array(onp.random.rand(6, 5, 3))
    n = vector_norm(v, axis=-1)
    self.assertAllClose(n, np.sqrt(np.sum(v * v, axis=2)), True)
    
    n = vector_norm(v, axis=1)
    self.assertAllClose(n, np.sqrt(np.sum(v * v, axis=1)), True)
    
    self.assertAllClose(0.0, vector_norm([]), False)
    self.assertAllClose(1.0, vector_norm([1]), False)


  def test_unit_vector(self):
    v0 = random_vector(3)
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

