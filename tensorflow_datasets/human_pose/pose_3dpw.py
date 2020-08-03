from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import pickle
import re

import numpy as np
import tensorflow as tf
from absl import logging

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.human_pose.mpii import annot_utils
from tensorflow_datasets.human_pose.mpii import skeleton
from tensorflow_datasets.core import lazy_imports
from tensorflow_datasets.core.download import extractor

NUM_JOINTS = 24

CITATION = """\
@inproceedings{andriluka14cvpr,
        author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
        title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2014},
        month = {June}
}"""


class Pose3DPWConfig(tfds.core.BuilderConfig):
  """BuilderConfig for '3DPW'.

  Args:
    people_included_from_two_actor_sequences (int): is either 0, 1 or 2.
      for 0: uses only sequences with one actor
      for 1: uses all sequences, but only extracts data of one actor
      for 2: uses only sequences with two actors.
      This config is necessary since datashapes with unknown number of frames and unknown
      number of actors are not allowed.
  """

  def __init__(self, people_included_from_two_actor_sequences=1, **kwargs):
    super(Pose3DPWConfig, self).__init__(**kwargs)
    self.people_included_from_two_actor_sequences = people_included_from_two_actor_sequences


class Pose3DPW(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('1.0.0')

  BUILDER_CONFIGS = [
    Pose3DPWConfig(
      name='all_sequences',
      description='A single actor pose from all sequences.',
      version='1.0.0',
      people_included_from_two_actor_sequences=1,
    ),
    Pose3DPWConfig(
      name='only_single_actor',
      description='Only sequences with one actor.',
      version='1.0.0',
      people_included_from_two_actor_sequences=0,
    ),
    Pose3DPWConfig(
      name='only_two_actors',
      description='Only sequences with both actors, including both poses.',
      version='1.0.0',
      people_included_from_two_actor_sequences=2,
    ),
  ]

  def _info(self):
    # data shape is dependent on the number of actors
    if self.builder_config.people_included_from_two_actor_sequences == 2:
      ACTORS = 2
    else:
      ACTORS = 1
    return tfds.core.DatasetInfo(
      builder=self,
      description="Human-annotated 2D human poses on in-the-wild images",
      features=tfds.features.FeaturesDict({
          'cam_intrinsics': tfds.features.Tensor(shape=(3, 3), dtype=tf.float64),
          "sex": tfds.features.Sequence(tfds.features.Text(), length=ACTORS),  # original 3DPW key: genders
          'smpl_betas': tfds.features.Tensor(shape=(ACTORS, 10), dtype=tf.float64),
          'smpl_betas_clothed': tfds.features.Tensor(shape=(ACTORS, 10), dtype=tf.float64),
          "frames": tfds.features.Sequence(tfds.features.FeaturesDict({
              "image": tfds.features.Image(encoding_format="jpeg"),
              "filename": tfds.features.Text(),
              "frame_sec": tfds.features.Tensor(shape=(), dtype=tf.float32),
              # different joint attributes
              "joints": tfds.features.Tensor(shape=(ACTORS, NUM_JOINTS, 3), dtype=tf.float64),  # original 3DPW key: jointPositions
              "center": tfds.features.Tensor(shape=(ACTORS, 3,), dtype=tf.float64),  # original 3DPW key: trans
              'smpl_poses': tfds.features.Tensor(shape=(ACTORS, 72), dtype=tf.float64),
              # additional attributes
              'cam_pose': tfds.features.Tensor(shape=(4, 4), dtype=tf.float64),
              'campose_valid': tfds.features.Tensor(shape=(ACTORS, ), dtype=tf.bool),
            })),
      }),
      homepage="http://human-pose.mpi-inf.mpg.de/",
      citation=CITATION)

  def _split_generators(self, dl_manager):
    paths = dl_manager.download({
      "images": "https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip",
      "sequences": "https://virtualhumans.mpi-inf.mpg.de/3DPW/sequenceFiles.zip",
    })
    extracted = dl_manager.extract(paths)
    sequences = path.join(extracted['sequences'], 'sequenceFiles')

    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs=dict(
          image_dir=path.join(extracted["images"], "imageFiles"),
          sequences=[path.join(sequences, 'train', x) for x in tf.io.gfile.listdir(path.join(sequences, 'train'))],
        )),
      tfds.core.SplitGenerator(
        name=tfds.Split.VALIDATION,
        gen_kwargs=dict(
          image_dir=path.join(extracted["images"], "imageFiles"),
          sequences=[path.join(sequences, 'validation', x) for x in tf.io.gfile.listdir(path.join(sequences, 'validation'))],
        )),
      tfds.core.SplitGenerator(
        name=tfds.Split.TEST,
        gen_kwargs=dict(
          image_dir=path.join(extracted["images"], "imageFiles"),
          sequences=[path.join(sequences, 'test', x) for x in tf.io.gfile.listdir(path.join(sequences, 'test'))],
        )),
    ]

  def _generate_examples(self, image_dir, sequences):
    if self.builder_config.people_included_from_two_actor_sequences == 2:
      ACTORS = 2
    else:
      ACTORS = 1

    def data_at_frame(data, idx):
      """Gathers data from frame idx for both 2 or 1 actor datashapes."""
      if isinstance(data, list):
        # one list item for each actor
        return np.stack(data)[:ACTORS, idx]
      return data[idx]

    for seq_filename in sequences:
      # collect data for all frames
      frames = []
      with open(seq_filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
      n_actors_in_sequence = len(data['genders'])
      if self.builder_config.people_included_from_two_actor_sequences == 0 \
          and n_actors_in_sequence == 2:
        continue
      elif self.builder_config.people_included_from_two_actor_sequences == 2 \
          and n_actors_in_sequence == 1:
        continue

      # clean data of betas, sometimes arrays have trailing 0s
      betas = [x[:10] for x in data['betas']]
      betas_clothed = [x[:10] for x in data['betas_clothed']]
      image_folder = path.join(image_dir, data['sequence'])
      for filename in sorted(tf.io.gfile.listdir(image_folder)):
        frame_idx = _get_frame_from_image(filename)
        frames.append({
          'image': path.join(image_folder, filename),
          'filename': filename,
          'frame_sec': np.float32(1.0 / 30 * frame_idx),  # video is 30 fps
          'joints': data_at_frame(data['jointPositions'], frame_idx).reshape([ACTORS, 24, 3]),
          'center': data_at_frame(data['trans'], frame_idx),
          'smpl_poses': data_at_frame(data['poses'], frame_idx),
          'cam_pose': data_at_frame(data['cam_poses'], frame_idx),
          'campose_valid': data_at_frame(data['campose_valid'], frame_idx).astype(bool),
        })
      features = {
          'smpl_betas': np.stack(betas)[:ACTORS],
          'smpl_betas_clothed': np.stack(betas_clothed)[:ACTORS],
          'sex': data['genders'][:ACTORS],
          'cam_intrinsics': data['cam_intrinsics'],
          'frames': frames,
      }
      yield str(data['sequence']), features

# Helper functions

IMAGE_FILE_RE = re.compile(r'image_(.+)\.jpg')


def _get_frame_from_image(filename):
  """Returns the frame index of an image file.

  Used to associate an image file
  with its corresponding label.
  Example:
    'bonn_000001_000019_leftImg8bit' -> 'bonn_000001_000019'

  Args:
    filename: image file name.

  Returns:
    idx of the associated frame.
  """
  return int(IMAGE_FILE_RE.match(filename).group(1))