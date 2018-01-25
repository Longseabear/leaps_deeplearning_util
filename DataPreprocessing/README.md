# Data Preprocessing
Utilities related to data preprocessing.
It contains the process of preprocessing a large image of imagenet.

# Layered_sampling.py
- Object
### Suppose you want to use only a small number of samples when the labeled data is contained in each label folder.
The easiest way is to load all the data, shuffle it, and load as much data as you want.
However, this is statistically incorrect data sampling.
If the luck is bad, some labels may not be included in the sample.
This function picks up the data evenly according to the number of labels.

# Using
python3 layered_sampling.py [--dataset_dir ] [--output_dir] [--file_name] [options]

# arguments
parser.add_argument('--dataset_dir', type=str,
                    help='Path to dataset directory.')
parser.add_argument('--output_dir', type=str,
                    help='Path to output directory.')
parser.add_argument('--file_name', type=str,
                    help='Path to output directory.')
parser.add_argument('--example_per_file', type=int, default=10000,
                    help='Number of Example Per Tfrecords files')
parser.add_argument('--iter_per_log', type=int, default=1000,
                    help='iter per log')
parser.add_argument('--object_dataset_num', type=int, default=100000,
                    help='total number of object datas')
parser.add_argument('--seed', type=int, default=13,
                    help='random_seed')
parser.add_argument('--corrupt_check', type=bool, default=False,
                    help='corrupt_check function')

# Additional explanation
This code converts the JPEG image into a TFrecords file.
If you modify the features of line 63, you can save it as TFrecords as you like.
Corrupt Check is used to select Corrupted image in imagenet 2001 version.

Tune in your mind as you like.

