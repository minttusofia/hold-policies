import os

import numpy as np

from .simple_sampler import SimpleSampler


class GoalImageSampler(SimpleSampler):

    def __init__(self, task, goal_image=None, **kwargs):
        super(GoalImageSampler, self).__init__(**kwargs)
        self._task = task
        self._num_steps = 2 if 'Image48MetaworldDrawerOpen' in task else 1
        self._goal_image_name = goal_image

    def _get_goal_image(self, stage=None):
        TOP_DIR = os.environ['HOLD_TOP_DIR']
        rlv_goal_image_dir = os.path.join(TOP_DIR, 'goal_images/rlv')
        dvd_goal_image_dir = os.path.join(TOP_DIR, 'goal_images/dvd')
        robodesk_goal_image_dir = os.path.join(TOP_DIR, 'goal_images/robodesk')
        if self._task == 'Image48MetaworldDrawerOpenSparse2D-v1':
            goal_image_prefix = (
                self._goal_image_name if self._goal_image_name
                else 'drawer_goal_image')
            if stage is None:
                stage = 2
            path = os.path.join(rlv_goal_image_dir,
                                f'{goal_image_prefix}_step{stage}.npy')
            goal_image = np.load(path)
            print('Setting goal to', path)
        elif self._task == 'Image48HumanLikeSawyerPushForwardEnv-v1':
            goal_image = np.load(
                os.path.join(
                    rlv_goal_image_dir,
                    f'{self._goal_image_name}.npy' if self._goal_image_name
                    else 'pushing_goal_image.npy'))
        elif self._task in ['CloseDrawerEnv1-v0', 'CloseDrawerEnv1-v1']:
            goal_image = np.load(
                os.path.join(dvd_goal_image_dir, 'task5.npy'))
        elif self._task in ['CupForwardEnv1-v0', 'CupForwardEnv1-v1']:
            goal_image = np.load(
                os.path.join(dvd_goal_image_dir, 'task41.npy'))
        elif self._task in ['FaucetRightEnv1-v0', 'FaucetRightEnv1-v1']:
            goal_image = np.load(
                os.path.join(dvd_goal_image_dir, 'task93.npy'))
        elif self._task in ['push_green', 'push_green_dense']:
            goal_image = np.load(
                os.path.join(robodesk_goal_image_dir, 'push_green.npy'))
        elif self._task in ['open_drawer', 'open_drawer_dense']:
            goal_image = np.load(
                os.path.join(robodesk_goal_image_dir, 'open_drawer.npy'))
        elif self._task in ['upright_block_off_table',
                            'upright_block_off_table_dense']:
            goal_image = np.load(
                os.path.join(robodesk_goal_image_dir,
                             'upright_block_off_table.npy'))
        else:
            raise ValueError(f'Goal image not defined for task {self._task}')
        print(f'goal image scale: [{np.min(goal_image)}, {np.max(goal_image)}]')
        return goal_image
