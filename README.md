# Medical-classifier
We implement deep learning classifier on lung CT scans to classify whether the person is sick or not, and get F1scores, recall, precision, auc, accuracy
--- 
## requirement
- python >= 3.6
- pytorch >= 1.0
- tensorboard
- sklearn

## introduce
- the io of our data is special, so if you want to use your own dataset on this repos, you should write your own data io to fit the Xinguan class in such ways:
  - you should provide img path list [string] and target list [1 or 0] 
  - the order of img path list and target list should be corresponded
- you can run train.py after reading and changing the forehead part of it, which is about the dataset 
- you can run test after you get your own weights file like this
> test.py --pretrained_weights checkpoints/your_weights_file

## Note:
We release this repos because we think some part of codes can be referenced, we do not recommand you to modify it directly for your usage because the modifying of data io may be complex
