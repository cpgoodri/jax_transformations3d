from transformations import *


def test_identity_matrix():
  I = identity_matrix()
  assert(np.allclose(I, np.dot(I, I)))
  assert(np.allclose(np.array([np.sum(I), np.trace(I)]), np.array([4.0, 4.0])))
  assert(np.allclose(I, np.identity(4)))
  print('identity_matrix() passed')


def test_translation_matrix():
  v = np.array(onp.random.random(3)) - 0.5
  assert(np.allclose(v, translation_matrix(v)[:3, 3]))
  print('translation_matrix() passed')


def test_translation_from_matrix():
  v0 = np.array(onp.random.random(3)) - 0.5
  v1 = translation_from_matrix(translation_matrix(v0))
  assert(np.allclose(v0, v1))
  print('translation_from_matrix() passed')









test_identity_matrix()
test_translation_matrix()
test_translation_from_matrix()


