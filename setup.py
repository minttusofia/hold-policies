from distutils.core import setup
from setuptools import find_packages


NAME = 'hold_policies'
VERSION = '0.0.1'
DESCRIPTION = (
    "RL with videos is an algorithm for leveraging observational"
    " data to speed the training of RL algorithms.  It is built on top of "
    "Softlearning, a deep reinforcement learning toolbox for training"
    " maximum entropy policies in continuous domains.")


setup(
    name=NAME,
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version=VERSION,
    description=DESCRIPTION,
    long_description=open('./README.md').read(),
    author='Karl Schmeckpeper',
    author_email='karls@seas.upenn.edu',
    url='https://github.com/kschmeckpeper/hold_policies',
    keywords=(
        'rl-with-videos',
        'rlv',
        'reinforcement-learning-with-videos',
        'softlearning',
        'soft-actor-critic',
        'sac',
        'soft-q-learning',
        'sql',
        'machine-learning',
        'reinforcement-learning',
        'deep-learning',
        'python',
    ),
    entry_points={
        'console_scripts': (
            'hold_policies=rl_with_videos.scripts.console_scripts:main',
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
