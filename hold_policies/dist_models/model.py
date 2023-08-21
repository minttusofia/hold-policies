"""Distance model base classes."""


class DistanceFn:
  """Given a state and a goal, returns a distance."""

  def __init__(
      self,
      distance_fn,
      history_length):
    self._distance_fn = distance_fn
    self.history_length = history_length

  def __call__(self, state, goal):
    return self._distance_fn(state, goal)


class EmbeddingFn:
  """Given a state and a goal, returns a distance in an embedding space."""

  def __init__(
      self,
      embedding_fn,
      distance_fn,
      history_length):
    self._embedding_fn = embedding_fn
    self._distance_fn = distance_fn
    self.history_length = history_length

  def __call__(self, state, goal):
    state_emb = self._embedding_fn(state)
    goal_emb = self._embedding_fn(goal)
    return self._distance_fn(state_emb, goal_emb)
