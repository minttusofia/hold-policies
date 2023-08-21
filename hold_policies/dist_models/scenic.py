import functools
import importlib
import jax
import jax.numpy as jnp

from scenic.projects.func_dist import pretrain_utils
from scenic.projects.func_dist import holdc_model

from hold_policies.dist_models import model as dist_model


class ScenicDistanceModel(dist_model.DistanceFn):

    def __init__(self, ckpt_path, scenic_config_path, distance_type='eucl',
                 distance_params=None):
        config = self._load_distance_model_config(scenic_config_path)
        model, train_state, _ = pretrain_utils.restore_model(config, ckpt_path)
        distance_fn, history_length = self._load_scenic_distance_fn(
            model, train_state, config, distance_type, distance_params)
        super().__init__(distance_fn, history_length)

    def _load_distance_model_config(self, scenic_config_path):
        scenic_config_path = scenic_config_path.replace('/', '.')
        scenic_config = importlib.import_module(scenic_config_path).get_config()
        return scenic_config

    def _load_scenic_distance_fn(
        self,
        model,
        train_state,
        config,
        distance_type,
        distance_params):
      """Load a scenic distance function based on model def and train state."""

      def zero_centre(frames):
        return frames * 2.0 - 1.0

      def central_crop_frames(frames):
        _, h, w, _ = frames.shape
        min_dim = min(h, w)
        margin_h = int((h - min_dim) / 2)
        margin_w = int((w - min_dim) / 2)
        cropped_frames = frames[:,
                                margin_h:margin_h + min_dim,
                                margin_w:margin_w + min_dim]
        return cropped_frames

      def apply_distance_model(
          model,
          train_state,
          config,
          state,
          goal,
        ):

        variables = {
            'params': train_state.optimizer['target'],
            **train_state.model_state
        }
        goal = jnp.expand_dims(goal, axis=0)
        # Expands the stacking axis if needed.
        state = jnp.reshape(state, state.shape[:3] + (-1,))
        # Frame stacking wrapper stacks on the last dimension.
        state = jnp.transpose(state, [3, 0, 1, 2])
        state = central_crop_frames(state)
        goal = central_crop_frames(goal)
        inputs = jnp.concatenate([state, goal], axis=0)
        inputs = resize_fn(inputs)
        if config.dataset_configs.zero_centering:
          inputs = zero_centre(inputs)
        # Add batch dimension.
        inputs = jnp.expand_dims(inputs, axis=0)
        dist = model.flax_model.apply(
            variables, inputs, train=False, mutable=False)
        dist = dist[0][0]
        return dist

      def apply_embedding_model(
          model,
          train_state,
          config,
          distance_type,
          distance_params,
          state,
          goal,
        ):

        variables = {
            'params': train_state.optimizer['target'],
            **train_state.model_state
        }
        goal = jnp.expand_dims(goal, axis=0)
        # Expands the stacking axis if needed.
        state = jnp.reshape(state, state.shape[:3] + (-1,))
        # Frame stacking wrapper stacks on the last dimension.
        state = jnp.transpose(state, [3, 0, 1, 2])
        state = central_crop_frames(state)
        goal = central_crop_frames(goal)
        inputs = jnp.concatenate([state, goal], axis=0)
        inputs = resize_fn(inputs)
        if config.dataset_configs.zero_centering:
          inputs = zero_centre(inputs)
        emb = model.flax_model.apply(
            variables, inputs, train=False, mutable=False)
        if distance_type == 'eucl':
            dist = jnp.linalg.norm(emb[0] - emb[1])
        if distance_type == 'sq_eucl':
            dist = jnp.square(jnp.linalg.norm(emb[0] - emb[1]))
        if distance_type == 'sq_eucl_huber':
            assert len(distance_params) == 3, (
                "3 parameters are needed for sq_eucl_huber.")
            alpha, beta, gamma = distance_params
            sq_eucl = jnp.square(jnp.linalg.norm(emb[0] - emb[1]))
            dist = alpha * sq_eucl + beta * jnp.sqrt(gamma + sq_eucl)
        return dist

      crop_size = config.dataset_configs.crop_size
      input_shape = (crop_size, crop_size, 3)
      resize_fn = jax.vmap(
          functools.partial(
              jax.image.resize, shape=input_shape, method='bilinear'),
          axis_name='time')

      if isinstance(model, holdc_model.TemporalContrastiveModel):
        distance_fn = functools.partial(
            apply_embedding_model, model, train_state, config, distance_type,
            distance_params)
        num_stacked_frames = 1
      else:
        distance_fn = functools.partial(
            apply_distance_model, model, train_state, config)
        num_stacked_frames = config.dataset_configs.num_frames - 1
      distance_fn = jax.jit(distance_fn)
      return distance_fn, num_stacked_frames

