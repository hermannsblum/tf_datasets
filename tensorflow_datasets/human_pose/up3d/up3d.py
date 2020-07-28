from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os import path
import re

import numpy as np
import tensorflow as tf

import tensorflow_datasets.public_api as tfds
from tensorflow_datasets.human_pose.mpii import skeleton

NUM_JOINTS = skeleton.s16.num_joints

CITATION = """\
@inproceedings{Lassner:UP:2017,
  title = {Unite the People: Closing the Loop Between 3D and 2D Human Representations},
  author = {Lassner, Christoph and Romero, Javier and Kiefel, Martin and Bogo, Federica and Black, Michael J. and Gehler, Peter V.},
  booktitle = {IEEE Conf. on Computer Vision and Pattern Recognition (CVPR)},
  month = jul,
  year = {2017},
  url = {http://up.is.tuebingen.mpg.de},
  month_numeric = {7}
}
"""


class Up3D(tfds.core.GeneratorBasedBuilder):
  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
      builder=self,
      description="3D human poses on in-the-wild images",
      features=tfds.features.FeaturesDict({
        "image": tfds.features.Image(encoding_format="png"),
        "filename": tfds.features.Text(),
        # different joint attributes
        "joints": tfds.features.Tensor(shape=(None, NUM_JOINTS, 3), dtype=tf.int64),
        "render_light": tfds.features.Image(encoding_format="png"),
      }),
      homepage="http://up.is.tuebingen.mpg.de",
      citation=CITATION)

  def _split_generators(self, dl_manager):
    paths = dl_manager.download({
      "3d": "http://files.is.tuebingen.mpg.de/classner/up/datasets/up-3d.zip",
      "p14": "http://files.is.tuebingen.mpg.de/classner/up/datasets/up-p14.zip"
    })
    extracted = dl_manager.extract(paths)
    up3d = path.join(extracted['3d'], 'up-3d')
    upp14 = path.join(extracted['p14'], 'p14_joints')

    def get_all_file_paths(sample_id):
      return {
        'image': path.join(up3d, '{}_image.png'.format(sample_id)),
        'joints': path.join(upp14, '{}_joints.npy'.format(sample_id)),
        'joint_annotation': path.join(up3d, '{}_joints.npy'.format(sample_id)),
        'render_ligt': path.join(up3d, '{}_render_light.png'.format(sample_id)),
      }

    return [
      tfds.core.SplitGenerator(
        name=tfds.Split.TRAIN,
        gen_kwargs=dict(samples=[
          get_all_file_paths(_get_id_from_image_name(x))
          for x in tf.io.gfile.GFile(path.join(up3d, 'train.txt')).readlines()
        ]))
    ]

  def _generate_examples(self, samples):
    for sample in samples:
      print(np.load(sample['joints']))
      print(np.load(sample['joint_annotation']))

      break
      features = {
        'image': sample['image'],
        'base_light': sample['base_light'],
      }
      return sample, features

IMAGE_FILEPATH_RE = re.compile(r'/(.+)_image.png')


def _get_id_from_image_name(image_name):
  """
  Returns the id of a data sample given the image file path as specified
  in the splits train.txt, val.txt or test.txt.
  """
  return IMAGE_FILEPATH_RE.match(image_name).group(1)
