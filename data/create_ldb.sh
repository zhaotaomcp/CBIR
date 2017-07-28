#!/usr/bin/env sh
# 将图片转存到leveldb中
#设定训练和测试数据list存储文件目录，编译好的caffe可执行文件目录
EXAMPLE=/opt/down/caffe-master/examples/cbir/data
DATA=/opt/down/caffe-master/examples/cbir/data
TOOLS=/opt/down/caffe-master/build/tools

#设定训练和测试图片集所在根目录
TRAIN_DATA_ROOT=/opt/down/caffe-master/examples/cbir/data/256/
VAL_DATA_ROOT=/opt/down/caffe-master/examples/cbir/data/256/

# 设定RESIZE=true将scale图片至256x256.如果图片已经做过resize，设定为false.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet validation data is stored."
  exit 1
fi

echo "Creating train leveldb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    --backend="leveldb" \
    / \
    $DATA/train.txt \
    $EXAMPLE/cbir_train_leveldb

echo "done"
