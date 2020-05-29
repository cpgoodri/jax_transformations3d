from transformations import *
from transformations import _AXES2TUPLE

from absl.testing import absltest
from absl.testing import parameterized

from jax import test_util as jtu
from jax import jit, grad, vmap

from jax.config import config as jax_config
jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS


from scipy.spatial.transform import Rotation as Rot

def _convert_quaternion(q):
  """Convert a scalar-first quaternion to a scalar-last quaternion
      This is used for testing with scipy.spatial.transform.Rotation
  """
  return onp.array([q[1],q[2],q[3],q[0]])

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


  def test_shear_matrix(self):
    angle = (onp.random.random() - 0.5) * 4*math.pi
    direct = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    normal = np.cross(direct, random_vector(3))
    S = shear_matrix(angle, direct, point, normal)
    self.assertAllClose(1, np.linalg.det(S), False)

  """
  def test_jit_shear_matrix(self):
    angle = (onp.random.random() - 0.5) * 4*math.pi
    direct = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    normal = np.cross(direct, random_vector(3))
    jshear_matrix = jit(shear_matrix)
    S0 = jshear_matrix(angle, direct, point, normal)
    S1 =  shear_matrix(angle, direct, point, normal)
    print(S0)
    print(S1)
    assert(is_same_transform(S0,S1))
  """

    

  def test_shear_from_matrix(self):
    angle = (onp.random.random() - 0.5) * 4*math.pi
    direct = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    normal = np.cross(direct, random_vector(3))
    S0 = shear_matrix(angle, direct, point, normal)
    angle, direct, point, normal = shear_from_matrix(S0)
    S1 = shear_matrix(angle, direct, point, normal)
    assert(is_same_transform(S0, S1))


  def test_euler_matrix(self):
    """
    R = euler_matrix(1, 2, 3, 'syxz')
    self.assertAllClose(np.sum(R[0]), -1.34786452, False)
    
    R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    self.assertAllClose(np.sum(R[0]), -0.383436184, False)
    
    ai, aj, ak = (4*math.pi) * (numpy.random.random(3) - 0.5)
    for axes in _AXES2TUPLE.keys():
       R = euler_matrix(ai, aj, ak, axes)
    for axes in _TUPLE2AXES.keys():
       R = euler_matrix(ai, aj, ak, axes)
    """
    def convert(s):
      if(s[0]=='r'):
        return s[1:].upper()
      return s[1:]
    axes_list = list(_AXES2TUPLE.keys())
    tuple_list = [_AXES2TUPLE[a] for a in axes_list]
    scipy_axes_list = [convert(s) for s in axes_list]

    num_rand = 2
    for _ in range(num_rand):
      angles = 2*np.pi*onp.random.rand(3)
      for a,t,s in zip(axes_list, tuple_list, scipy_axes_list):
        Ma = euler_matrix(angles[0], angles[1], angles[2], a)
        Mt = euler_matrix(angles[0], angles[1], angles[2], t)
        Ms = index_update(np.identity(4), index[:3, :3], np.array(Rot.from_euler(s, angles).as_matrix()))
        assert(is_same_transform(Ma, Mt))
        assert(is_same_transform(Ma, Ms))

  def test_jit_euler_matrix(self):
    R1 = euler_matrix(1, 2, 3, 'syxz')
    R2 = jit(euler_matrix,static_argnums=3)(1, 2, 3, 'syxz')
    assert(is_same_transform(R1,R2))
    

  def test_euler_from_matrix(self):
    R0 = euler_matrix(1, 2, 3, 'syxz')
    al, be, ga = euler_from_matrix(R0, 'syxz')
    R1 = euler_matrix(al, be, ga, 'syxz')
    self.assertAllClose(R0, R1, True)
    
    angles = (4*math.pi) * (onp.random.random(3) - 0.5)
    for axes in _AXES2TUPLE.keys():
      R0 = euler_matrix(axes=axes, *angles)
      R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
      self.assertAllClose(R0, R1, True)
    for axes in _AXES2TUPLE.values():
      R0 = euler_matrix(axes=axes, *angles)
      R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
      self.assertAllClose(R0, R1, True)

  def test_jit_euler_from_matrix(self):
    R0 = euler_matrix(1, 2, 3, 'szxz')
    al, be, ga = jit(euler_from_matrix, static_argnums=1)(R0, 'szxz')
    R1 = euler_matrix(al, be, ga, 'szxz')
    self.assertAllClose(R0, R1, True)



  def test_euler_from_quaternion(self):
    q = random_quaternion()
    R1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
    for axes in _AXES2TUPLE.keys():
      al, be, ga = euler_from_quaternion(q, axes)
      R0 = euler_matrix(al, be, ga, axes)
      self.assertAllClose(np.array(R0[:3,:3]), np.array(R1), True)

  def test_jit_euler_from_quaternion(self):
    q = random_quaternion()
    jeuler_from_quaternion = jit(euler_from_quaternion, static_argnums=1)

    axes = ['sxyz', 'szxz', 'rxyz', 'rzxz']
    for a in axes:
      angles0 = euler_from_quaternion(q, a)
      angles1 = jeuler_from_quaternion(q, a)
      self.assertAllClose(angles0, angles1, True)


  def test_quaternion_from_euler(self):
    def convert(s):
      if(s[0]=='r'):
        return s[1:].upper()
      return s[1:]
    axes_list = list(_AXES2TUPLE.keys())
    scipy_axes_list = [convert(s) for s in axes_list]

    angles = 2*np.pi*onp.random.rand(3)
    for a,s in zip(axes_list, scipy_axes_list):
      q0 = quaternion_from_euler(angles[0], angles[1], angles[2], a)
      q1 = Rot.from_euler(s, angles).as_quat()
      self.assertAllClose(_convert_quaternion(q0), q1, True)
      #self.assertAllClose(q, np.array([0.435953, 0.310622, -0.718287, 0.444435]), True)

  def test_jit_quaternion_from_euler(self):
    angles = 2*np.pi*onp.random.rand(3)
    jquaternion_from_euler = jit(quaternion_from_euler, static_argnums=3)

    axes = ['sxyz', 'szxz', 'rxyz', 'rzxz']
    for a in axes:
      q0 = quaternion_from_euler(angles[0], angles[1], angles[2], a)
      q1 = jquaternion_from_euler(angles[0], angles[1], angles[2], a)
      self.assertAllClose(q0, q1, True)


  def test_quaternion_about_axis(self):
    q0 = quaternion_about_axis(0.123, [1, 0, 0])
    q1 = Rot.from_rotvec(0.123*onp.array([1, 0, 0])).as_quat()
    self.assertAllClose(_convert_quaternion(q0), q1, True)
    #numpy.allclose(q, [0.99810947, 0.06146124, 0, 0])

  def test_jit_quaternion_about_axis(self):
    q0 = quaternion_about_axis(0.123, [1, 0, 0])
    q1 = jit(quaternion_about_axis)(0.123, [1, 0, 0])
    self.assertAllClose(q0, q1, True)


  def test_quaternion_matrix(self):
    #M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    q = np.array([0.998109470983817859, 0.061461239268365025, 0, 0])
    M0 = quaternion_matrix(q)
    M1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
    self.assertAllClose(M0, rotation_matrix(0.123, [1, 0, 0]), True)
    self.assertAllClose(M0[:3,:3], M1, False)
   
    q = np.array([1, 0, 0, 0])
    M0 = quaternion_matrix(q)
    M1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
    self.assertAllClose(M0, np.identity(4), True)
    self.assertAllClose(M0[:3,:3], M1, False)
    
    q = np.array([0, 1, 0, 0])
    M0 = quaternion_matrix(q)
    M1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
    self.assertAllClose(M0, np.diag(np.array([1, -1, -1, 1], dtype=np.float64)), True)
    self.assertAllClose(M0[:3,:3], M1, False)

    for _ in range(5):
      q = random_quaternion()
      M0 = quaternion_matrix(q)[:3,:3]
      M1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
      self.assertAllClose(M0, M1, False)











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


  def test_vector_product(self):
    v = vector_product([2, 0, 0], [0, 3, 0])
    self.assertAllClose(v, np.array([0, 0, 6]), False)
    v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    v1 = [[3], [0], [0]]
    v = vector_product(v0, v1)
    self.assertAllClose(v, np.array([[0, 0, 0, 0], [0, 0, 6, 6], [0, -6, 0, -6]]), False)
    v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    v = vector_product(v0, v1, axis=1)
    self.assertAllClose(v, np.array([[0, 0, 6], [0, -6, 0], [6, 0, 0], [0, -6, 6]]), False)


  def angle_between_vectors(self):
    a = angle_between_vectors([1, -2, 3], [-1, 2, -3])
    self.assertAllClose(a, math.pi, False)
    
    a = angle_between_vectors([1, -2, 3], [-1, 2, -3], directed=False)
    self.assertAllClose(a, 0, False)
    
    v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    v1 = [[3], [0], [0]]
    a = angle_between_vectors(v0, v1)
    self.assertAllClose(a, np.array([0, 1.5708, 1.5708, 0.95532]), True)
    
    v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    a = angle_between_vectors(v0, v1, axis=1)
    self.assertAllClose(a, np.array([1.5708, 1.5708, 1.5708, 0.95532]), True)


  def test_inverse_matrix(self):
    M0 = random_rotation_matrix()
    M1 = inverse_matrix(M0.T)
    self.assertAllClose(M1, np.linalg.inv(M0.T), True)
    
    for size in range(1, 7):
      M0 = np.array(onp.random.rand(size, size))
      M1 = inverse_matrix(M0)
      self.assertAllClose(M1, np.linalg.inv(M0), True)



  def test_concatenate_matrices(self):
    M = np.array(onp.random.rand(16).reshape((4, 4)) - 0.5)
    self.assertAllClose(M, concatenate_matrices(M), True)
    self.assertAllClose(np.dot(M, M.T), concatenate_matrices(M, M.T), True)




  def assert_is_same_transform(self, matrix0, matrix1):
    matrix0 = np.array(matrix0, dtype=np.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = np.array(matrix1, dtype=np.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return self.assertAllClose(matrix0, matrix1, False)







if __name__ == '__main__':
  absltest.main()

