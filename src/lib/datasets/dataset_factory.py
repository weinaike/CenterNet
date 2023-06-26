from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# from .sample.ctdet_point import CTDetPointDataset

from .point import PointOTF

dataset_factory = {
  'point':PointOTF
}
  
