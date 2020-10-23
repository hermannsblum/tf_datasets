"""youtube_boundingboxes dataset."""

import tensorflow_datasets.public_api as tfds
import tensorflow as tf
import csv
from pytube import YouTube
from pytube.exceptions import RegexMatchError
from collections import defaultdict
from time import sleep

cv2 = tfds.core.lazy_imports.cv2

# TODO(youtube_boundingboxes): BibTeX citation
_CITATION = """
"""

_DESCRIPTION = """
YouTube-BoundingBoxes is a large-scale data set with densely-sampled
high-quality single-object bounding box annotations.
All video segments were human-annotated with high precision classifications
and bounding boxes at 1 frame per second.
"""


class YoutubeBoundingboxes(tfds.core.GeneratorBasedBuilder):
  """TODO(youtube_boundingboxes): Short description of my dataset."""

  # TODO(youtube_boundingboxes): Set up version.
  VERSION = tfds.core.Version('0.1.0')

  def _info(self):
    # TODO(youtube_boundingboxes): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        # This is the description that will appear on the datasets page.
        description=_DESCRIPTION,
        # tfds.features.FeatureConnectors
        features=tfds.features.FeaturesDict(
            {
                'video':
                    tfds.features.Sequence(
                        tfds.features.FeaturesDict(
                            {
                                'frame':
                                    tfds.features.Image(
                                        shape=[None, None, 3],
                                        encoding_format='jpeg'
                                    ),
                                'bbox':
                                    tfds.features.BBoxFeature(),
                                'timestamp_ms':
                                    tf.int64,
                                'object_present':
                                    tf.bool,
                            }
                        )
                    ),
                'class':
                    tfds.features.ClassLabel(num_classes=23),
                'youtube_id':
                    tfds.features.Text(),
            }
        ),
        # Homepage of the dataset for documentation
        homepage='https://research.google.com/youtube-bb/index.html',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""
    # TODO(youtube_boundingboxes): Downloads the data and defines the splits
    # dl_manager is a tfds.download.DownloadManager that can be used to
    # download and extract URLs
    files = dl_manager.download_and_extract(
        {
            'train_objects':
                'https://research.google.com/youtube-bb/yt_bb_detection_train.csv.gz',
            'test_objects':
                'https://research.google.com/youtube-bb/yt_bb_detection_validation.csv.gz',
        }
    )

    video_urls = {}

    def process_csv(csvreader):
      # groups annotations by object identifier
      annotations = defaultdict(list)
      for row in csvreader:
        if row[0] not in video_urls:
          try:
            sleep(2)  # otherwise yt rate-limiting kicks in
            yt = YouTube('youtube.com/watch?v={}'.format(row[0]))
            video_urls[row[0]] = str(
                yt.streams.filter(file_extension='mp4'
                                 ).get_highest_resolution().url
            )
          except KeyError:
            # video does not exists anymore
            continue
          except RegexMatchError:
            continue
        # create a unique object identifier from youtube, class, and object id
        object_identifier = '{}_{}_{}'.format(row[0], row[2], row[4])
        annotations[object_identifier].append(
            {
                'youtube_id':
                    row[0],
                'timestamp_ms':
                    int(row[1]),
                'class_id':
                    int(row[2]),
                'class_name':
                    row[3],
                'object_id':
                    row[4],
                'object_presence':
                    bool(row[5]),
                'bounding_box':
                    tfds.features.BBox(
                        xmin=float(row[6]),
                        xmax=float(row[7]),
                        ymin=float(row[8]),
                        ymax=float(row[9])
                    )
            }
        )
      return annotations

    with open(files['train_objects'], newline='') as f:
      csvreader = csv.reader(f, delimiter=',')
      train_objects = process_csv(csvreader)
    with open(files['test_objects'], newline='') as f:
      csvreader = csv.reader(f, delimiter=',')
      test_objects = process_csv(csvreader)
    videos = dl_manager.download_and_extract(video_urls)

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                'videos': videos,
                'objects': train_objects
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                'videos': videos,
                'objects': test_objects
            },
        ),
    ]

  def _generate_examples(self, videos, objects):
    """Yields examples."""
    for key, detections in objects.items():
      # sort detections by timestamp
      detections = sorted(detections, key=lambda x: x['timestamp_ms'])
      video = cv2.VideoCapture(videos[detections[0]['youtube_id']])
      n_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
      fps = int(video.get(cv2.CAP_PROP_FPS))
      #TODO check fps
      video_features = []
      for detection in detections:
        # go to timestamp
        video.set(cv2.CAP_PROP_POS_MSEC, detection['timestamp_ms'])
        success, frame = video.read()
        # convert BGR -> RGB
        frame = frame[..., ::-1]
        if not success:
          print(
              'VIDEO {} COULD NOT BE READ CORRECTLY'.format(
                  detection['youtube_id']
              )
          )
          break
        video_features.append(
            {
                'frame': frame,
                'bbox': detection['bounding_box'],
                'timestamp_ms': detection['timestamp_ms'],
                'object_present': detection['object_presence'],
            }
        )
      if len(detections) != len(video_features):
        # could not extract all frames
        break
      all_features = {
          'video': video_features,
          'youtube_id': detections[0]['youtube_id'],
          'class': detections[0]['class_id'],
      }
      yield key, all_features
