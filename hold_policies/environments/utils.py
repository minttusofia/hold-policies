from .adapters.gym_adapter import GymAdapter
from .adapters.robodesk_adapter import RoboDeskAdapter

ADAPTERS = {
    'gym': GymAdapter,
}

try:
    from .adapters.dm_control_adapter import DmControlAdapter
    ADAPTERS['dm_control'] = DmControlAdapter
except ModuleNotFoundError as e:
    if 'dm_control' not in e.msg:
        raise

    print("Warning: dm_control package not found. Run"
          " `pip install git+https://github.com/deepmind/dm_control.git`"
          " to use dm_control environments.")

try:
    from .adapters.robosuite_adapter import RobosuiteAdapter
    ADAPTERS['robosuite'] = RobosuiteAdapter
except ModuleNotFoundError as e:
    if 'robosuite' not in e.msg:
        raise

    print("Warning: robosuite package not found. Run `pip install robosuite`"
          " to use robosuite environments.")

UNIVERSES = set(ADAPTERS.keys())

def get_environment(universe, domain, task, environment_params):
    return ADAPTERS[universe](domain, task, **environment_params)

def get_environment_from_params(environment_params):
    universe = environment_params['universe']
    task = environment_params['task']
    domain = environment_params['domain']
    environment_kwargs = environment_params.get('kwargs', {}).copy()

    return get_environment(universe, domain, task, env_params)

def get_goal_example_environment_from_variant(variant, evaluation=False):
    import gym
    task = variant['task']
    domain = variant['domain']
    if domain == 'robodesk':
        import robodesk
        reward = 'dense' if '_dense' in task and not evaluation else 'sparse'
        task = task.replace('_dense', '')
        env = RoboDeskAdapter(robodesk.RoboDesk(task=task, reward=reward))
        return GymAdapter(env=env, goal_based=variant['goal_based'])

    if evaluation and 'Dense' in task:
        task = task.replace('Dense', '')

    if task not in [env.id for env  in gym.envs.registry.all()]:
        if 'Manip' in task:
            import manip_envs
        elif variant['domain'] == 'tabletop':
            import sim_env.tabletop
        else:
            from multiworld.envs.mujoco import register_goal_example_envs
            register_goal_example_envs()
            from metaworld.envs.mujoco import register_rl_with_videos_custom_envs
            register_rl_with_videos_custom_envs()
#            import mj_envs.hand_manipulation_suite

    #        from metaworld.envs.mujoco.sawyer_xyz import register_environments; register_environments()
    return GymAdapter(env=gym.make(task), goal_based=variant['goal_based'])
