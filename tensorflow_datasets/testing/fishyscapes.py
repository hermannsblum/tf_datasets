"""Script to generate fake Fishyscapes data."""
import tensorflow as tf
from zipfile import ZipFile

from tensorflow_datasets.testing.cityscapes import generate_ids, create_zipfile
from tensorflow_datasets.testing.fake_data_utils import get_random_png

if __name__ == '__main__':
  example_dir = ('tensorflow_datasets/testing/test_data/fake_examples/fishyscapes')
  base_path = example_dir + '/{}.zip'
  # generate image ids matching between zipfiles
  train_ids = [*generate_ids('01_Turmstr_17'),
               *generate_ids('02_Goethe_Str_6')]
  test_ids = list(generate_ids('03_Schlossallee_1'))
  splits = {'train': train_ids, 'test': test_ids}
  with tf.Graph().as_default():
    create_zipfile(base_path.format('leftImg8bit'),
                   splits_with_ids=splits,
                   suffixes=['leftImg8bit'])
    with ZipFile(base_path.format('mask'), 'w') as z:
        for idx, img_id in enumerate((*train_ids, *test_ids)[::2]):
            img = get_random_png(height=1024, width=2048, channels=1)
            z.write(img, '{:04d}_{}_labels.png'.format(idx, img_id))
