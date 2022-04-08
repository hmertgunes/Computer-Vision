import os

def mkdir(p):
  if not os.path.exists(p):
    os.mkdir(p)

def link(src, dst):
  if not os.path.exists(dst):
    os.symlink(src, dst, target_is_directory=True)

mkdir('../VGG/large_files/fruits-360-small')

# Choosing some pictures as limited dataset
classes = [
  'apple_golden_1',
  'carrot_1',
  'cabbage_white_1',
  'cucumber_1',
  'zucchini_1',
  'pear_3',
  'eggplant_violet_1',
  'apple_red_delicious_1'
]

# Transporting(copy) files which we chose from large to small 
train_path_from = os.path.abspath('../VGG/large_files/fruits-360/Training')
valid_path_from = os.path.abspath('../VGG/large_files/fruits-360/Validation')

train_path_to = os.path.abspath('../VGG/large_files/fruits-360-small/Training')
valid_path_to = os.path.abspath('../VGG/large_files/fruits-360-small/Validation')

mkdir(train_path_to)
mkdir(valid_path_to)


for c in classes:
  link(train_path_from + '/' + c, train_path_to + '/' + c)
  link(valid_path_from + '/' + c, valid_path_to + '/' + c)