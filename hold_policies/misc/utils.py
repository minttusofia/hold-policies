import collections
import datetime
import os
import random

import tensorflow as tf
import numpy as np


PROJECT_PATH = os.path.dirname(
    os.path.realpath(os.path.join(__file__, '..', '..')))


DEFAULT_SNAPSHOT_MODE = 'none'
DEFAULT_SNAPSHOT_GAP = 1000


def initialize_tf_variables(session, only_uninitialized=True):
    variables = tf.global_variables() + tf.local_variables()

    def is_initialized(variable):
        try:
            session.run(variable)
            return True
        except tf.errors.FailedPreconditionError:
            return False

        return False

    if only_uninitialized:
        variables = [
            variable for variable in variables
            if not is_initialized(variable)
        ]

    session.run(tf.variables_initializer(variables))


def set_seed(seed):
    seed %= 4294967294
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    print("Using seed {}".format(seed))


def datetimestamp(divider='-', datetime_divider='T'):
    now = datetime.datetime.now()
    return now.strftime(
        '%Y{d}%m{d}%dT%H{d}%M{d}%S'
        ''.format(d=divider, dtd=datetime_divider))


def datestamp(divider='-'):
    return datetime.date.today().isoformat().replace('-', divider)


def timestamp(divider='-'):
    now = datetime.datetime.now()
    time_now = datetime.datetime.time(now)
    return time_now.strftime(
        '%H{d}%M{d}%S'.format(d=divider))


def concat_obs_z(obs, z, num_skills):
    """Concatenates the observation to a one-hot encoding of Z."""
    assert np.isscalar(z)
    z_one_hot = np.zeros(num_skills)
    z_one_hot[z] = 1
    return np.hstack([obs, z_one_hot])


def split_aug_obs(aug_obs, num_skills):
    """Splits an augmented observation into the observation and Z."""
    (obs, z_one_hot) = (aug_obs[:-num_skills], aug_obs[-num_skills:])
    z = np.where(z_one_hot == 1)[0][0]
    return (obs, z)


def _make_dir(filename):
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)


def save_video(video_frames, filename):
    import cv2
    _make_dir(filename)

    video_frames = np.flip(video_frames, axis=-1)

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'MPNG')
    fps = 30.0
    (height, width, _) = video_frames[0].shape
    writer = cv2.VideoWriter(filename, fourcc, fps, (width, height))
#    for i in range(len(video_frames)):
#        cv2.imwrite(filename+"_test_im{}.png".format(i), video_frames[i])
    for video_frame in video_frames:
        writer.write(video_frame)

    writer.release()


def deep_update(d, *us):
    d = d.copy()

    for u in us:
        u = u.copy()
        for k, v in u.items():
            d[k] = (
                deep_update(d.get(k, {}), v)
                if isinstance(v, collections.Mapping)
                else v)

    return d


def get_git_rev(path=PROJECT_PATH, search_parent_directories=True):
    try:
        import git
        from git.exc import InvalidGitRepositoryError
    except ImportError:
        print(
            "Warning: gitpython not installed."
            " Unable to log git rev."
            " Run `pip install gitpython` if you want git revs to be logged.")
        return None

    try:
        repo = git.Repo(
            path, search_parent_directories=search_parent_directories)
        git_rev = repo.active_branch.commit.name_rev
    except TypeError:
        git_rev = repo.head.object.name_rev
    except InvalidGitRepositoryError:
        git_rev = None

    return git_rev


def flatten(unflattened, parent_key='', separator='.'):
    items = []
    for k, v in unflattened.items():
        if separator in k:
            raise ValueError(
                "Found separator ({}) from key ({})".format(separator, k))
        new_key = parent_key + separator + k if parent_key else k
        if isinstance(v, collections.MutableMapping) and v:
            items.extend(flatten(v, new_key, separator=separator).items())
        else:
            items.append((new_key, v))

    return dict(items)


def unflatten(flattened, separator='.'):
    result = {}
    for key, value in flattened.items():
        parts = key.split(separator)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result

def mixup(features, labels, alpha=0.2):
    def unison_shuffled_copies(a, b):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        return a[p], b[p]

    features_2, labels_2 = unison_shuffled_copies(features, labels)
    _lambda = np.random.beta(alpha, alpha, size=(features.shape[0], 1))
    features_convex = _lambda*features + (1 - _lambda)*features_2
    labels_convex = _lambda*labels + (1 - _lambda)*labels_2

    return features_convex, labels_convex
