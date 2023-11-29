---

layout: post
title: 梳理mmdetection3d数据处理代码
category: 计算机视觉
tags: mmdetection3d hid
keywords: 3D detection
typora-root-url: ../..
---

* TOC
{:toc}
## 前期准备

点云数据需要使用`bin`格式储存，标注信息使用`txt`文件储存，可以使用[LabelCloud](https://github.com/ch-sa/labelCloud)进行3d标注。

## 基本数据格式

**mmdetection3d**支持多种预设的数据格式，如**kitti, Lyft, nus, s3dis, scannet, semantickitti, sunrgbd, waymo**等。在进行自定义数据任务时，由于**mmdetection3d**的自定义任务仍在升级中，因此目前推荐使用kitti的预设数据格式并作出对应的修改。

自定义数据集的训练有两种方式：

### 使用预设的kitti数据集

[官方kitti数据集说明文档](https://mmdetection3d.readthedocs.io/zh-cn/latest/advanced_guides/datasets/kitti.html)

文件夹结构按如下方式组织：

```python
mmdetection3d
├── mmdet3d
├── tools
├── configs
├── data
│   ├── kitti # 此文件夹名可自定义
│   │   ├── ImageSets
│   │   ├── testing
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── velodyne
│   │   ├── training
│   │   │   ├── calib
│   │   │   ├── image_2
│   │   │   ├── label_2
│   │   │   ├── velodyne
│   │   │   ├── planes (optional)
```



**calib**与**image_2**文件不参与训练，它们的文件名与点云文件velodyne与标注文件**label**一一对应仅在后处理时3d检测框投影到2d图像上发挥作用，因此**calib**与image_2文件可以使用**fake file**。

可以使用以下代码生成calib文件

```python
import os

def generate_calib_files(input_folder, output_folder, custom_content):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的txt文件
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            # 构建输出文件的完整路径
            output_filepath = os.path.join(output_folder, filename)

            # 写入新的txt文件
            with open(output_filepath, 'w') as output_file:
                output_file.write(custom_content.lstrip('\n'))

            print(f"生成了新的calib文件：{output_filepath}")

# 输入文件夹路径
input_folder_path = 'kitti_labels'

# 输出文件夹路径
output_folder_path = 'calib'

# 自定义内容
custom_content = '''
P0: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 0.000000000000e+00 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.797842000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P2: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 4.575831000000e+01 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 -3.454157000000e-01 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 4.981016000000e-03
P3: 7.070493000000e+02 0.000000000000e+00 6.040814000000e+02 -3.341081000000e+02 0.000000000000e+00 7.070493000000e+02 1.805066000000e+02 2.330660000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 3.201153000000e-03
R0_rect: 9.999128000000e-01 1.009263000000e-02 -8.511932000000e-03 -1.012729000000e-02 9.999406000000e-01 -4.037671000000e-03 8.470675000000e-03 4.123522000000e-03 9.999556000000e-01
Tr_velo_to_cam: 6.927964000000e-03 -9.999722000000e-01 -2.757829000000e-03 -2.457729000000e-02 -1.162982000000e-03 2.749836000000e-03 -9.999955000000e-01 -6.127237000000e-02 9.999753000000e-01 6.931141000000e-03 -1.143899000000e-03 -3.321029000000e-01
Tr_imu_to_velo: 9.999976000000e-01 7.553071000000e-04 -2.035826000000e-03 -8.086759000000e-01 -7.854027000000e-04 9.998898000000e-01 -1.482298000000e-02 3.195559000000e-01 2.024406000000e-03 1.482454000000e-02 9.998881000000e-01 -7.997231000000e-01
'''
# 生成calib文件
generate_calib_files(input_folder_path, output_folder_path, custom_content)

```

ImageSets文件夹中的文件写入了划分数据集的文件名，如下所示：

```
# train.txt
000000
000001
000002
000003
```

然后通过以下代码预处理数据集并生成pkl文件

```python
python tools/create_data.py kitti --root-path ./data/kitti --out-dir ./data/kitti --extra-tag kitti
```

处理后的文件夹结构形式为：

```
kitti
├── ImageSets
│   ├── test.txt
│   ├── train.txt
│   ├── trainval.txt
│   ├── val.txt
├── testing
│   ├── calib
│   ├── image_2
│   ├── velodyne
│   ├── velodyne_reduced
├── training
│   ├── calib
│   ├── image_2
│   ├── label_2
│   ├── velodyne
│   ├── velodyne_reduced
│   ├── planes (optional)
├── kitti_gt_database
│   ├── xxxxx.bin
├── kitti_infos_train.pkl
├── kitti_infos_val.pkl
├── kitti_dbinfos_train.pkl
├── kitti_infos_test.pkl
├── kitti_infos_trainval.pkl
```

### 使用自定义数据集来进行训练（更新中）

需要增加或修改的文件如下：

```
mmdetection3d
├── configs
│   ├── _base_
│   │   ├── datasets
│   │   │   ├── kitti-hid.py #文件名可自定义
├── mmdet
│   ├── configs
│   │   ├── _base_
│   │   │   ├── datasets
│   │   │   │   ├── kitti-hid.py #文件名可自定义
│   ├── datasets
│   │   ├── __init__.py
│   │   ├── hid_datasets.py #文件名可自定义
├── tools
│   ├── dataset_converters
│   │   ├── creat_gt_database.py
│   │   ├── hid_converter.py
│   │   ├── update_info_to_v2.py
│   │   │   ├── datasets
│   │   │   │   ├── kitti-hid.py #文件名可自定义
│   ├── create_data.py
```

上述文件储存在github的mmdetection3d库中
