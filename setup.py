from distutils.core import setup
from setuptools import find_packages


NAME = 'hold_policies'
VERSION = '0.0.1'
DESCRIPTION = (
    "HOLD policies implements the training of RL policies using "
    "learned reward models, based on the SAC implementation and "
    "manipulation environments of RL with videos.")

setup(
    name=NAME,
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    version=VERSION,
    description=DESCRIPTION,
    long_description=open('./README.md').read(),
    author='Minttu Alakuijala',
    author_email='minttu.alakuijala@inria.fr',
    url='https://github.com/minttusofia/hold-policies',
    keywords=(
        'reward-models',
        'rl-with-videos',
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
            'hold_policies=hold_policies.scripts.console_scripts:main',
        )
    },
    requires=(),
    zip_safe=True,
    license='MIT'
)
