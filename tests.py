from jax_transformations3d import *
from jax_transformations3d import _AXES2TUPLE

from absl.testing import absltest
from absl.testing import parameterized

from jax import test_util as jtu
from jax import jit, grad, vmap, random

from jax.config import config as jax_config

jax_config.parse_flags_with_absl()
FLAGS = jax_config.FLAGS

from scipy.spatial.transform import Rotation as Rot


def _convert_quaternion(q):
  """Convert a scalar-first quaternion to a scalar-last quaternion
      This is used for testing with scipy.spatial.transform.Rotation
  """
  return onp.array([q[1], q[2], q[3], q[0]])


def _check_scipy_quaternion_sign(q):
  if (q[3] < 0.0):
    return -q
  return q


class TransformationsTest(jtu.JaxTestCase):

  def test_identity_matrix(self):
    I = identity_matrix()
    self.assertAllClose(I, jnp.dot(I, I))
    self.assertAllClose(
        jnp.array([jnp.sum(I), jnp.trace(I)]), jnp.array([4.0, 4.0]))
    self.assertAllClose(I, jnp.identity(4))

  def test_jit_identity_matrix(self):
    I0 = identity_matrix()
    I1 = jit(identity_matrix)()
    self.assertAllClose(I0, I1)

  def test_translation_matrix(self):
    v = random_vector(3) - 0.5
    self.assertAllClose(v, translation_matrix(v)[:3, 3])

  def test_jit_translation_matrix(self):
    v = random_vector(3) - 0.5
    T0 = translation_matrix(v)
    T1 = jit(translation_matrix)(v)
    self.assertAllClose(T0, T1)

  def test_translation_from_matrix(self):
    v0 = random_vector(3) - 0.5
    v1 = translation_from_matrix(translation_matrix(v0))
    self.assertAllClose(v0, v1)

  def test_jit_translation_from_matrix(self):
    v0 = random_vector(3) - 0.5
    T = translation_matrix(v0)
    v1 = translation_from_matrix(T)
    v2 = jit(translation_from_matrix)(T)
    self.assertAllClose(v1, v2)

  def test_reflection_matrix(self):
    v0 = onp.random.random(4) - 0.5
    v0[3] = 1.
    v0 = jnp.array(v0)
    v1 = random_vector(3) - 0.5
    R = reflection_matrix(v0, v1)
    self.assertAllClose(2, jnp.trace(R), check_dtypes=False)
    self.assertAllClose(v0, jnp.dot(R, v0))
    v2 = onp.array(v0).copy()
    v3 = onp.array(v0).copy()
    v2[:3] += v1
    v3[:3] -= v1
    v2 = jnp.array(v2)
    v3 = jnp.array(v3)
    self.assertAllClose(v2, jnp.dot(R, v3))

  def test_jit_reflection_matrix(self):
    v0 = onp.random.random(4) - 0.5
    v0[3] = 1.
    v0 = jnp.array(v0)
    v1 = random_vector(3) - 0.5
    R0 = reflection_matrix(v0, v1)
    R1 = jit(reflection_matrix)(v0, v1)
    self.assertAllClose(R0, R1)

  def test_reflection_from_matrix(self):
    v0 = random_vector(3) - 0.5
    v1 = random_vector(3) - 0.5
    M0 = reflection_matrix(v0, v1)
    point, normal = reflection_from_matrix(M0)
    M1 = reflection_matrix(point, normal)
    assert (is_same_transform(M0, M1))

  """
  def test_jit_reflection_from_matrix(self):
    v0 = random_vector(3) - 0.5
    v1 = random_vector(3) - 0.5
    M0 = reflection_matrix(v0, v1)
    point0, normal0 = reflection_from_matrix(M0)
    point1, normal1 = jit(reflection_from_matrix)(M0)
    self.assertAllClose(point0, point1)
    self.assertAllClose(normal0, normal1)
  """

  def test_rotation_matrix(self):
    R = rotation_matrix(math.pi / 2, jnp.array([0, 0, 1]), jnp.array([1, 0, 0]))
    self.assertAllClose(
        jnp.dot(R, jnp.array([0, 0, 0, 1])),
        jnp.array([1, -1, 0, 1]),
        check_dtypes=False)

    angle = (onp.random.random() - 0.5) * (2 * math.pi)
    direc = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    R0 = rotation_matrix(angle, direc, point)
    R1 = rotation_matrix(angle - 2 * math.pi, direc, point)
    assert (is_same_transform(R0, R1))

    R0 = rotation_matrix(angle, direc, point)
    R1 = rotation_matrix(-angle, -direc, point)
    assert (is_same_transform(R0, R1))

    I = jnp.identity(4, jnp.float64)
    self.assertAllClose(I, rotation_matrix(math.pi * 2, direc))

    self.assertAllClose(
        2,
        jnp.trace(rotation_matrix(math.pi / 2, direc, point)),
        check_dtypes=False)

  def test_jit_rotation_matrix(self):
    angle = (onp.random.random() - 0.5) * (2 * math.pi)
    direc = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    R0 = rotation_matrix(angle, direc)
    R1 = jit(rotation_matrix)(angle, direc)
    self.assertAllClose(R0, R1)

    R0 = rotation_matrix(angle, direc, point)
    R1 = jit(rotation_matrix)(angle, direc, point)
    self.assertAllClose(R0, R1)

  def test_rotation_from_matrix(self):
    angle = (onp.random.random() - 0.5) * (2 * math.pi)
    direc = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    R0 = rotation_matrix(angle, direc, point)
    angle, direc, point = rotation_from_matrix(R0)
    R1 = rotation_matrix(angle, direc, point)
    assert (is_same_transform(R0, R1))

  """
  def test_jit_rotation_from_matrix(self):
    angle = (onp.random.random() - 0.5) * (2*math.pi)
    direc = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    R0 = rotation_matrix(angle, direc, point)
    angle0, direc0, point0 = rotation_from_matrix(R0)
    angle1, direc1, point1 = jit(rotation_from_matrix)(R0)
    self.assertAllClose(angle0, angle1)
    self.assertAllClose(direc0, direc1)
    self.assertAllClose(point0, point1)
  """

  def test_scale_matrix(self):
    v = (onp.random.rand(4, 5) - 0.5) * 20
    v[3] = 1
    v = jnp.array(v)
    S = scale_matrix(-1.234)
    self.assertAllClose(jnp.dot(S, v)[:3], -1.234 * v[:3])

    factor = onp.random.random() * 10 - 5
    origin = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    S = scale_matrix(factor, origin)
    S = scale_matrix(factor, origin, direct)

  def test_jit_scale_matrix(self):
    factor = onp.random.random() * 10 - 5
    origin = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    S0 = scale_matrix(factor, origin)
    S1 = jit(scale_matrix)(factor, origin)
    self.assertAllClose(S0, S1)

    S0 = scale_matrix(factor, origin, direct)
    S1 = jit(scale_matrix)(factor, origin, direct)
    self.assertAllClose(S0, S1)

  def test_scale_from_matrix(self):
    factor = onp.random.random() * 10 - 5
    origin = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    S0 = scale_matrix(factor, origin)
    factor, origin, direction = scale_from_matrix(S0)
    S1 = scale_matrix(factor, origin, direction)
    assert (is_same_transform(S0, S1))

    S0 = scale_matrix(factor, origin, direct)
    factor, origin, direction = scale_from_matrix(S0)
    S1 = scale_matrix(factor, origin, direction)
    assert (is_same_transform(S0, S1))

  """
  def test_jit_scale_from_matrix(self):
    factor = onp.random.random() * 10 - 5
    origin = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    S0 = scale_matrix(factor, origin, direct)
    factor0, origin0, direction0 = scale_from_matrix(S0)
    factor0, origin0, direction0 = jit(scale_from_matrix)(S0)
    self.assertAllClose(factor0, factor1)
    self.assertAllClose(origin0, origin1)
    self.assertAllClose(direction0, direction1)
  """

  def test_projection_matrix(self):
    P = projection_matrix([0, 0, 0], [1, 0, 0])
    self.assertAllClose(P[1:, 1:], jnp.identity(4)[1:, 1:])
    point = random_vector(3) - 0.5
    normal = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    persp = random_vector(3) - 0.5
    P0 = projection_matrix(point, normal)
    P1 = projection_matrix(point, normal, direction=direct)
    P2 = projection_matrix(point, normal, perspective=persp)
    P3 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    assert (is_same_transform(P2, jnp.dot(P0, P3)))
    P = projection_matrix([3, 0, 0], [1, 1, 0], [1, 0, 0])
    v0 = (onp.random.rand(4, 5) - 0.5) * 20
    v0[3] = 1
    v0 = jnp.array(v0)
    v1 = jnp.dot(P, v0)
    self.assertAllClose(v1[1], v0[1])
    self.assertAllClose(v1[0], 3 - v1[1])

  def test_jit_projection_matrix(self):
    point = random_vector(3) - 0.5
    normal = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    persp = random_vector(3) - 0.5
    P0 = projection_matrix(point, normal)
    P1 = jit(projection_matrix)(point, normal)
    self.assertAllClose(P0, P1)

    P0 = projection_matrix(point, normal, direction=direct)
    P1 = jit(projection_matrix)(point, normal, direction=direct)
    self.assertAllClose(P0, P1)

    P0 = projection_matrix(point, normal, perspective=persp)
    P1 = jit(projection_matrix)(point, normal, perspective=persp)
    self.assertAllClose(P0, P1)

    P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    #For some reason, static_argnums is necessary here but not for the previous examples
    P1 = jit(
        projection_matrix, static_argnums=(2, 4))(point, normal, None, persp,
                                                  True)
    self.assertAllClose(P0, P1)

  def test_projection_from_matrix(self):
    point = random_vector(3) - 0.5
    normal = random_vector(3) - 0.5
    direct = random_vector(3) - 0.5
    persp = random_vector(3) - 0.5
    P0 = projection_matrix(point, normal)
    result = projection_from_matrix(P0)
    P1 = projection_matrix(*result)
    assert (is_same_transform(P0, P1))
    P0 = projection_matrix(point, normal, direct)
    result = projection_from_matrix(P0)
    P1 = projection_matrix(*result)
    assert (is_same_transform(P0, P1))
    P0 = projection_matrix(point, normal, perspective=persp, pseudo=False)
    result = projection_from_matrix(P0, pseudo=False)
    P1 = projection_matrix(*result)
    assert (is_same_transform(P0, P1))
    P0 = projection_matrix(point, normal, perspective=persp, pseudo=True)
    result = projection_from_matrix(P0, pseudo=True)
    P1 = projection_matrix(*result)
    assert (is_same_transform(P0, P1))

  def test_shear_matrix(self):
    angle = (onp.random.random() - 0.5) * 4 * math.pi
    direct = random_vector(3) - 0.5
    direct = unit_vector(direct)
    point = random_vector(3) - 0.5
    normal = jnp.cross(direct, random_vector(3))
    S = shear_matrix(angle, direct, point, normal)
    self.assertAllClose(
        1.0, jnp.linalg.det(S), check_dtypes=False, atol=1e-12, rtol=1e-12)

  """
  def test_jit_shear_matrix(self):
    angle = (onp.random.random() - 0.5) * 4*math.pi
    direct = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    normal = jnp.cross(direct, random_vector(3))
    jshear_matrix = jit(shear_matrix)
    S0 = jshear_matrix(angle, direct, point, normal)
    S1 =  shear_matrix(angle, direct, point, normal)
    print(S0)
    print(S1)
    assert(is_same_transform(S0,S1))
  """

  def test_shear_from_matrix(self):
    angle = (onp.random.random() - 0.5) * 4 * math.pi
    direct = random_vector(3) - 0.5
    point = random_vector(3) - 0.5
    normal = jnp.cross(direct, random_vector(3))
    S0 = shear_matrix(angle, direct, point, normal)
    angle, direct, point, normal = shear_from_matrix(S0)
    #print( angle, direct, point, normal)
    S1 = shear_matrix(angle, direct, point, normal)
    assert (is_same_transform(S0, S1))

  def test_euler_matrix(self):
    """
    R = euler_matrix(1, 2, 3, 'syxz')
    self.assertAllClose(jnp.sum(R[0]), -1.34786452, check_dtypes=False)
    
    R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    self.assertAllClose(jnp.sum(R[0]), -0.383436184, check_dtypes=False)
    
    ai, aj, ak = (4*math.pi) * (numpy.random.random(3) - 0.5)
    for axes in _AXES2TUPLE.keys():
       R = euler_matrix(ai, aj, ak, axes)
    for axes in _TUPLE2AXES.keys():
       R = euler_matrix(ai, aj, ak, axes)
    """

    def convert(s):
      if (s[0] == 'r'):
        return s[1:].upper()
      return s[1:]

    axes_list = list(_AXES2TUPLE.keys())
    tuple_list = [_AXES2TUPLE[a] for a in axes_list]
    scipy_axes_list = [convert(s) for s in axes_list]

    num_rand = 1
    for _ in range(num_rand):
      angles = 2 * jnp.pi * onp.random.rand(3)
      for a, t, s in zip(axes_list, tuple_list, scipy_axes_list):
        Ma = euler_matrix(angles[0], angles[1], angles[2], a)
        Mt = euler_matrix(angles[0], angles[1], angles[2], t)
        Ms = jnp.identity(4).at[:3, :3].set(
            jnp.array(Rot.from_euler(s, angles).as_matrix()))
        #Ms = index_update(jnp.identity(4), index[:3, :3], jnp.array(Rot.from_euler(s, angles).as_matrix()))
        assert (is_same_transform(Ma, Mt))
        assert (is_same_transform(Ma, Ms))

  def test_jit_euler_matrix(self):
    R1 = euler_matrix(1, 2, 3, 'syxz')
    R2 = jit(euler_matrix, static_argnums=3)(1, 2, 3, 'syxz')
    assert (is_same_transform(R1, R2))

  def test_euler_from_matrix(self):
    R0 = euler_matrix(1, 2, 3, 'syxz')
    al, be, ga = euler_from_matrix(R0, 'syxz')
    R1 = euler_matrix(al, be, ga, 'syxz')
    self.assertAllClose(R0, R1)

    angles = (4 * math.pi) * (onp.random.random(3) - 0.5)
    for axes in _AXES2TUPLE.keys():
      R0 = euler_matrix(axes=axes, *angles)
      R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
      self.assertAllClose(R0, R1)
    for axes in _AXES2TUPLE.values():
      R0 = euler_matrix(axes=axes, *angles)
      R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
      self.assertAllClose(R0, R1)

  def test_jit_euler_from_matrix(self):
    R0 = euler_matrix(1, 2, 3, 'szxz')
    al, be, ga = jit(euler_from_matrix, static_argnums=1)(R0, 'szxz')
    R1 = euler_matrix(al, be, ga, 'szxz')
    self.assertAllClose(R0, R1)

  def test_euler_from_quaternion(self):
    q = random_quaternion()
    R1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
    for axes in _AXES2TUPLE.keys():
      al, be, ga = euler_from_quaternion(q, axes)
      R0 = euler_matrix(al, be, ga, axes)
      self.assertAllClose(
          jnp.array(R0[:3, :3]), jnp.array(R1), atol=1e-12, rtol=1e-12)

  def test_jit_euler_from_quaternion(self):
    q = random_quaternion()
    jeuler_from_quaternion = jit(euler_from_quaternion, static_argnums=1)

    axes = ['sxyz', 'szxz', 'rxyz', 'rzxz']
    for a in axes:
      angles0 = euler_from_quaternion(q, a)
      angles1 = jeuler_from_quaternion(q, a)
      self.assertAllClose(angles0, angles1)

  def test_quaternion_from_euler(self):

    def convert(s):
      if (s[0] == 'r'):
        return s[1:].upper()
      return s[1:]

    axes_list = list(_AXES2TUPLE.keys())
    scipy_axes_list = [convert(s) for s in axes_list]

    angles = 2 * jnp.pi * onp.random.rand(3)
    for a, s in zip(axes_list, scipy_axes_list):
      q0 = quaternion_from_euler(angles[0], angles[1], angles[2], a)
      q1 = Rot.from_euler(s, angles).as_quat()
      self.assertAllClose(_convert_quaternion(q0), q1)
      #self.assertAllClose(q, jnp.array([0.435953, 0.310622, -0.718287, 0.444435]))

  def test_jit_quaternion_from_euler(self):
    angles = 2 * jnp.pi * onp.random.rand(3)
    jquaternion_from_euler = jit(quaternion_from_euler, static_argnums=3)

    axes = ['sxyz', 'szxz', 'rxyz', 'rzxz']
    for a in axes:
      q0 = quaternion_from_euler(angles[0], angles[1], angles[2], a)
      q1 = jquaternion_from_euler(angles[0], angles[1], angles[2], a)
      self.assertAllClose(q0, q1)

  def test_quaternion_about_axis(self):
    q0 = quaternion_about_axis(0.123, [1, 0, 0])
    q1 = Rot.from_rotvec(0.123 * onp.array([1, 0, 0])).as_quat()
    self.assertAllClose(_convert_quaternion(q0), q1)
    #numpy.allclose(q, [0.99810947, 0.06146124, 0, 0])

  def test_jit_quaternion_about_axis(self):
    q0 = quaternion_about_axis(0.123, [1, 0, 0])
    q1 = jit(quaternion_about_axis)(0.123, [1, 0, 0])
    self.assertAllClose(q0, q1)

  def test_quaternion_matrix(self):
    #M = quaternion_matrix([0.99810947, 0.06146124, 0, 0])
    q = jnp.array([0.998109470983817859, 0.061461239268365025, 0, 0])
    M0 = quaternion_matrix(q)
    M1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
    self.assertAllClose(M0, rotation_matrix(0.123, [1, 0, 0]))
    self.assertAllClose(M0[:3, :3], M1, check_dtypes=False)

    q = jnp.array([1, 0, 0, 0])
    M0 = quaternion_matrix(q)
    M1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
    self.assertAllClose(M0, jnp.identity(4))
    self.assertAllClose(M0[:3, :3], M1, check_dtypes=False)

    q = jnp.array([0, 1, 0, 0])
    M0 = quaternion_matrix(q)
    M1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
    self.assertAllClose(M0,
                        jnp.diag(jnp.array([1, -1, -1, 1], dtype=jnp.float64)))
    self.assertAllClose(M0[:3, :3], M1, check_dtypes=False)

    for _ in range(5):
      q = random_quaternion()
      M0 = quaternion_matrix(q)[:3, :3]
      M1 = Rot.from_quat(_convert_quaternion(q)).as_matrix()
      self.assertAllClose(M0, M1, check_dtypes=False)

  def test_jit_quaternion_matrix(self):
    jquaternion_matrix = jit(quaternion_matrix)
    for _ in range(5):
      q = random_quaternion()
      M0 = quaternion_matrix(q)
      M1 = jquaternion_matrix(q)
      self.assertAllClose(M0, M1)

  def test_quaternion_from_matrix(self):
    q = quaternion_from_matrix(jnp.identity(4))
    self.assertAllClose(q, jnp.array([1, 0, 0, 0], dtype=jnp.float64))

    q = quaternion_from_matrix(jnp.identity(4), False)
    self.assertAllClose(q, jnp.array([1, 0, 0, 0], dtype=jnp.float64))

    q = quaternion_from_matrix(
        jnp.diag(jnp.array([1, -1, -1, 1], dtype=jnp.float64)))
    assert (jnp.allclose(q, jnp.array([0, 1, 0, 0], dtype=jnp.float64)) or
            jnp.allclose(q, jnp.array([0, -1, 0, 0], dtype=jnp.float64)))

    R = rotation_matrix(0.123, (1, 2, 3))
    q0 = quaternion_from_matrix(R, True)
    q1 = _check_scipy_quaternion_sign(
        Rot.from_matrix(onp.array(R[:3, :3])).as_quat())
    self.assertAllClose(_convert_quaternion(q0), q1)
    #numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])

    R = jnp.array([[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
                   [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]],
                  dtype=jnp.float64)
    q0 = quaternion_from_matrix(R)
    #q1 = Rot.from_matrix(R[:3,:3]).as_quat()
    #self.assertAllClose(_convert_quaternion(q0), q1)
    self.assertAllClose(
        q0,
        jnp.array([0.19069, 0.43736, 0.87485, -0.083611]),
        atol=1e-5,
        rtol=1e-5)

    R = jnp.array([[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
                   [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]],
                  dtype=jnp.float64)
    q0 = quaternion_from_matrix(R)
    #q1 = Rot.from_matrix(R[:3,:3]).as_quat()
    #self.assertAllClose(_convert_quaternion(q0), q1)
    self.assertAllClose(
        q0,
        jnp.array([0.82336615, -0.13610694, 0.46344705, -0.29792603]),
        atol=1e-5,
        rtol=1e-5)

    R = random_rotation_matrix()
    q0 = quaternion_from_matrix(R)
    q1 = _check_scipy_quaternion_sign(
        Rot.from_matrix(onp.array(R[:3, :3])).as_quat())
    self.assertAllClose(_convert_quaternion(q0), q1)
    assert (is_same_transform(R, quaternion_matrix(q0)))

    assert (is_same_quaternion(
        quaternion_from_matrix(R, isprecise=False),
        quaternion_from_matrix(R, isprecise=True)))

    R = euler_matrix(0.0, 0.0, jnp.pi / 2.0)
    assert (is_same_quaternion(
        quaternion_from_matrix(R, isprecise=False),
        quaternion_from_matrix(R, isprecise=True)))

  def test_jit_quaternion_from_matrix(self):
    R = random_rotation_matrix()
    jquaternion_from_matrix = jit(quaternion_from_matrix, static_argnums=1)

    q0 = quaternion_from_matrix(R, True)
    q1 = jquaternion_from_matrix(R, True)
    self.assertAllClose(q0, q1)

    q0 = quaternion_from_matrix(R, False)
    q1 = jquaternion_from_matrix(R, False)
    self.assertAllClose(q0, q1)

  def test_quaternion_multiply(self):
    q = quaternion_multiply([4, 1, -2, 3], [8, -5, 6, 7])
    self.assertAllClose(q, jnp.array([28, -44, -14, 48], dtype=jnp.float64))

  def test_jit_quaternion_multiply(self):
    #q = jit(quaternion_multiply)(jnp.array([4, 1, -2, 3],dtype=jnp.float64), jnp.array([8, -5, 6, 7], dtype=jnp.float64))
    q = jit(quaternion_multiply)([4, 1, -2, 3], [8, -5, 6, 7])
    self.assertAllClose(q, jnp.array([28, -44, -14, 48], dtype=jnp.float64))

  def test_quaternion_conjugate(self):
    q0 = random_quaternion()
    q1 = quaternion_conjugate(q0)
    assert (q1[0] == q0[0])
    self.assertAllClose(q1[1:], -q0[1:])

  def test_jit_quaternion_conjugate(self):
    q0 = random_quaternion()
    q1 = jit(quaternion_conjugate)(q0)
    assert (q1[0] == q0[0])
    self.assertAllClose(q1[1:], -q0[1:])

  def test_quaternion_inverse(self):
    q0 = random_quaternion()
    q1 = quaternion_inverse(q0)
    self.assertAllClose(
        quaternion_multiply(q0, q1), jnp.array([1, 0, 0, 0], dtype=jnp.float64))

  def test_jit_quaternion_inverse(self):
    q0 = random_quaternion()
    q1 = jit(quaternion_inverse)(q0)
    self.assertAllClose(
        quaternion_multiply(q0, q1), jnp.array([1, 0, 0, 0], dtype=jnp.float64))

  def test_quaternion_real(self):
    self.assertAllClose(quaternion_real([3, 0, 1, 2]), 3.0, check_dtypes=False)

  def test_jit_quaternion_real(self):
    self.assertAllClose(
        jit(quaternion_real)(jnp.array([3, 0, 1, 2], dtype=jnp.float64)),
        3.0,
        check_dtypes=False)

  def test_quaternion_imag(self):
    self.assertAllClose(
        quaternion_imag([3, 0, 1, 2]), jnp.array([0., 1., 2.],
                                                 dtype=jnp.float64))

  def test_jit_quaternion_imag(self):
    self.assertAllClose(
        jit(quaternion_imag)([3, 0, 1, 2]),
        jnp.array([0., 1., 2.], dtype=jnp.float64))

  def test_random_quaternion(self):
    q = random_quaternion()
    self.assertAllClose(1.0, vector_norm(q), check_dtypes=False)
    assert (len(q.shape) == 1)
    assert (q.shape[0] == 4)

    q = random_quaternion(onp.random.random(3))
    self.assertAllClose(1.0, vector_norm(q), check_dtypes=False)
    assert (len(q.shape) == 1)
    assert (q.shape[0] == 4)

    key = random.PRNGKey(0)
    q = random_quaternion(key=key)
    self.assertAllClose(1.0, vector_norm(q), check_dtypes=False)
    assert (len(q.shape) == 1)
    assert (q.shape[0] == 4)

    rand = random.uniform(key, (3,), minval=0.0, maxval=1.0, dtype=jnp.float64)
    q = random_quaternion(rand)
    self.assertAllClose(1.0, vector_norm(q), check_dtypes=False)
    assert (len(q.shape) == 1)
    assert (q.shape[0] == 4)

  def test_jit_random_quaternion(self):
    key = random.PRNGKey(0)

    q = jit(random_quaternion)(key=key)
    self.assertAllClose(1.0, vector_norm(q), check_dtypes=False)
    assert (len(q.shape) == 1)
    assert (q.shape[0] == 4)

    rand = random.uniform(key, (3,), minval=0.0, maxval=1.0, dtype=jnp.float64)
    q = jit(random_quaternion)(rand)
    self.assertAllClose(1.0, vector_norm(q), check_dtypes=False)
    assert (len(q.shape) == 1)
    assert (q.shape[0] == 4)

  def test_random_rotation_matrix(self):
    R = random_rotation_matrix()
    self.assertAllClose(jnp.dot(R.T, R), jnp.identity(4))

    rand = onp.random.random(3)
    R = random_rotation_matrix(rand)
    self.assertAllClose(jnp.dot(R.T, R), jnp.identity(4))

    key = random.PRNGKey(0)
    rand = random.uniform(key, (3,), minval=0.0, maxval=1.0, dtype=jnp.float64)
    R = random_rotation_matrix(rand)
    self.assertAllClose(jnp.dot(R.T, R), jnp.identity(4))

  def test_jit_random_rotation_matrix(self):
    key = random.PRNGKey(0)
    rand = random.uniform(key, (3,), minval=0.0, maxval=1.0, dtype=jnp.float64)
    R = random_rotation_matrix(rand)
    self.assertAllClose(jnp.dot(R.T, R), jnp.identity(4))

  def test_vector_norm(self):
    v = random_vector(3)
    n = vector_norm(v)
    self.assertAllClose(n, jnp.linalg.norm(v))

    v = jnp.array(onp.random.rand(6, 5, 3))
    n = vector_norm(v, axis=-1)
    self.assertAllClose(n, jnp.sqrt(jnp.sum(v * v, axis=2)))

    n = vector_norm(v, axis=1)
    self.assertAllClose(n, jnp.sqrt(jnp.sum(v * v, axis=1)))

    self.assertAllClose(0.0, vector_norm([]), check_dtypes=False)
    self.assertAllClose(1.0, vector_norm([1]), check_dtypes=False)

  def test_jit_vector_norm(self):
    v = random_vector(3)
    n0 = vector_norm(v)
    n1 = jit(vector_norm)(v)
    self.assertAllClose(n0, n1)

    v = jnp.array(onp.random.rand(6, 5, 3))
    n0 = vector_norm(v, axis=-1)
    n1 = jit(vector_norm, static_argnums=1)(v, -1)
    self.assertAllClose(n0, n1)

  def test_unit_vector(self):
    v0 = random_vector(3)
    v1 = unit_vector(v0)
    self.assertAllClose(v1, v0 / jnp.linalg.norm(v0))
    self.assertAllClose(unit_vector([1]), jnp.array([1.0]))

    v0 = onp.random.rand(5, 4, 3)
    v1 = unit_vector(v0, axis=-1)
    v2 = v0 / jnp.expand_dims(jnp.sqrt(jnp.sum(v0 * v0, axis=2)), 2)
    self.assertAllClose(v1, v2)

    v1 = unit_vector(v0, axis=1)
    v2 = v0 / jnp.expand_dims(jnp.sqrt(jnp.sum(v0 * v0, axis=1)), 1)
    self.assertAllClose(v1, v2)

  def test_jit_unit_vector(self):
    v = random_vector(3)
    v0 = unit_vector(v)
    v1 = jit(unit_vector)(v)
    self.assertAllClose(v0, v1)

    v0 = onp.random.rand(5, 4, 3)
    v1 = unit_vector(v0, axis=-1)
    v2 = jit(unit_vector, static_argnums=1)(v0, -1)
    self.assertAllClose(v1, v2)

    v1 = unit_vector(v0, axis=1)
    v2 = jit(unit_vector, static_argnums=1)(v0, 1)
    self.assertAllClose(v1, v2)

  def test_random_vector(self):
    v = random_vector(10000)
    assert (jnp.all(v >= 0) and jnp.all(v < 1))
    v0 = random_vector(10)
    v1 = random_vector(10)
    assert (not any(v0 == v1))

    v = random_vector((10000, 5))
    assert (jnp.all(v >= 0) and jnp.all(v < 1))
    assert (v.shape == (10000, 5))

    key = random.PRNGKey(0)
    key, split1, split2, split3, split4 = random.split(key, 5)
    v = random_vector(10000, split1)
    assert (jnp.all(v >= 0) and jnp.all(v < 1))
    v0 = random_vector(10, split2)
    v1 = random_vector(10, split3)
    assert (not any(v0 == v1))

    v = random_vector((10000, 5), split4)
    assert (jnp.all(v >= 0) and jnp.all(v < 1))
    assert (v.shape == (10000, 5))

  """
  def test_jit_random_vector(self):
    jrandom_vector = jit(random_vector, static_argnums=1)
    key = random.PRNGKey(0)
    key, split1, split2, split3 = random.split(key,4)
    v = jrandom_vector((10000,), split1)
    assert(jnp.all(v >= 0) and jnp.all(v < 1))
    v0 = jrandom_vector((10,), split2)
    v1 = jrandom_vector((10,), split3)
    assert( not any(v0 == v1) )
  """

  def test_vector_product(self):
    v = vector_product([2, 0, 0], [0, 3, 0])
    self.assertAllClose(v, jnp.array([0, 0, 6]), check_dtypes=False)

    v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    v1 = [[3], [0], [0]]
    v = vector_product(v0, v1)
    self.assertAllClose(
        v,
        jnp.array([[0, 0, 0, 0], [0, 0, 6, 6], [0, -6, 0, -6]]),
        check_dtypes=False)

    v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    v = vector_product(v0, v1, axis=1)
    self.assertAllClose(
        v,
        jnp.array([[0, 0, 6], [0, -6, 0], [6, 0, 0], [0, -6, 6]]),
        check_dtypes=False)

  def test_jit_vector_product(self):
    v0 = vector_product([2, 0, 0], [0, 3, 0])
    v1 = jit(vector_product)([2, 0, 0], [0, 3, 0])
    self.assertAllClose(v0, v1)

    t0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    t1 = [[3], [0], [0]]
    v0 = vector_product(t0, t1)
    v1 = jit(vector_product)(t0, t1)
    self.assertAllClose(v0, v1)

    t0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    t1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    v0 = vector_product(t0, t1, axis=1)
    v1 = jit(vector_product, static_argnums=2)(t0, t1, 1)
    self.assertAllClose(v0, v1)

  def angle_between_vectors(self):
    a = angle_between_vectors([1, -2, 3], [-1, 2, -3])
    self.assertAllClose(a, math.pi, check_dtypes=False)

    a = angle_between_vectors([1, -2, 3], [-1, 2, -3], directed=False)
    self.assertAllClose(a, 0, check_dtypes=False)

    v0 = [[2, 0, 0, 2], [0, 2, 0, 2], [0, 0, 2, 2]]
    v1 = [[3], [0], [0]]
    a = angle_between_vectors(v0, v1)
    self.assertAllClose(a, jnp.array([0, 1.5708, 1.5708, 0.95532]))

    v0 = [[2, 0, 0], [2, 0, 0], [0, 2, 0], [2, 0, 0]]
    v1 = [[0, 3, 0], [0, 0, 3], [0, 0, 3], [3, 3, 3]]
    a = angle_between_vectors(v0, v1, axis=1)
    self.assertAllClose(a, jnp.array([1.5708, 1.5708, 1.5708, 0.95532]))

  def test_inverse_matrix(self):
    M0 = random_rotation_matrix()
    M1 = inverse_matrix(M0.T)
    self.assertAllClose(M1, jnp.linalg.inv(M0.T))

    for size in range(1, 7):
      M0 = jnp.array(onp.random.rand(size, size))
      M1 = inverse_matrix(M0)
      self.assertAllClose(M1, jnp.linalg.inv(M0))

  def test_concatenate_matrices(self):
    M = jnp.array(onp.random.rand(16).reshape((4, 4)) - 0.5)
    self.assertAllClose(M, concatenate_matrices(M))
    self.assertAllClose(jnp.dot(M, M.T), concatenate_matrices(M, M.T))

  def test_matrix_apply_and_quaternion_apply(self):
    key = random.PRNGKey(0)
    key, split1, split2 = random.split(key, 3)
    q = random_quaternion()
    M = quaternion_matrix(q)
    v3 = random.uniform(
        split1, (3,), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    v4 = jnp.pad(v3, (0, 1), mode='constant', constant_values=1.0)
    vm3 = random.uniform(
        split2, (10, 3), minval=-1.0, maxval=1.0, dtype=jnp.float64)
    vm4 = jnp.pad(vm3, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)

    for v in [v3, v4, vm3, vm4]:
      vrot_m = matrix_apply(M, v)
      vrot_q = quaternion_apply(q, v)
      self.assertAllClose(vrot_m, vrot_q)

  def test_quaternion_apply_shapes(self):
    key = random.PRNGKey(0)
    m = 10
    l = 6
    key, *splits = random.split(key, m + 1)
    qs = vmap(random_quaternion, in_axes=(None, 0))(None, jnp.array(splits))
    key, split = random.split(key)
    vs = random_vector((m, l), split)

    shape = quaternion_apply(qs[0], vs[0]).shape
    assert (shape == (l,))
    shape = quaternion_apply(qs, vs[0]).shape
    assert (shape == (m, l))
    shape = quaternion_apply(qs[0], vs).shape
    assert (shape == (m, l))
    shape = quaternion_apply(qs, vs).shape
    assert (shape == (m, l))

  def assert_is_same_transform(self, matrix0, matrix1):
    matrix0 = jnp.array(matrix0, dtype=jnp.float64, copy=True)
    matrix0 /= matrix0[3, 3]
    matrix1 = jnp.array(matrix1, dtype=jnp.float64, copy=True)
    matrix1 /= matrix1[3, 3]
    return self.assertAllClose(matrix0, matrix1, check_dtypes=False)


if __name__ == '__main__':
  absltest.main()
