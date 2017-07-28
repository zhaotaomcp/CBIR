#!/usr/bin/env sh
# 计算leveldb中存储的训练图片集的均值
# 请预先训练好caffe


/opt/down/caffe-master/build/tools/compute_image_mean -backend leveldb cbir_train_leveldb mycbir_mean.binaryproto

echo "Compute images mean file Done!"
