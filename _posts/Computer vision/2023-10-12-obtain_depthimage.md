---

layout: post
title: 深度图的训练流程
category: 计算机视觉
tags: detection
keywords: depth image
typora-root-url: ../..
---

* TOC
{:toc}


### 1. 硬件配置

#### 激光相机介绍

+ 公司：**LMI Technologies Inc** ｜ 荷兰TKH子公司

+ 系列：Gocator系列｜轮廓传感器

  <img src="/public/upload/camera/c1.png" style="zoom:50%;" />

+ 名称：Gocator 2450

#### 使用文档目录

[Gocator 产品参数和选型.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E4%BA%A7%E5%93%81%E5%8F%82%E6%95%B0%E5%92%8C%E9%80%89%E5%9E%8B.pdf)

[Gocator 内置工具简介.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E5%86%85%E7%BD%AE%E5%B7%A5%E5%85%B7%E7%AE%80%E4%BB%8B.pdf)

[Gocator 单传感器连接.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E5%8D%95%E4%BC%A0%E6%84%9F%E5%99%A8%E8%BF%9E%E6%8E%A5.pdf)

[Gocator 扫描数据保存.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E6%89%AB%E6%8F%8F%E6%95%B0%E6%8D%AE%E4%BF%9D%E5%AD%98.pdf)

[Gocator 界面概述及固件更新.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E7%95%8C%E9%9D%A2%E6%A6%82%E8%BF%B0%E5%8F%8A%E5%9B%BA%E4%BB%B6%E6%9B%B4%E6%96%B0.pdf)

[Gocator 线激光点云获取.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E7%BA%BF%E6%BF%80%E5%85%89%E7%82%B9%E4%BA%91%E8%8E%B7%E5%8F%96.pdf)

[Gocator 线激光轮廓获取.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E7%BA%BF%E6%BF%80%E5%85%89%E8%BD%AE%E5%BB%93%E8%8E%B7%E5%8F%96.pdf	)

[Gocator 输出与状态.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E8%BE%93%E5%87%BA%E4%B8%8E%E7%8A%B6%E6%80%81.pdf)

[LMI系列3D相机快速使用说明书.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E4%BA%A7%E5%93%81%E5%8F%82%E6%95%B0%E5%92%8C%E9%80%89%E5%9E%8B.pdf)

#### 硬件清单

+ windows PC ｜ 激光相机｜电源适配器｜网线

#### 硬件安装

传感器可以连接到计算机的以太网端口进 行设置，也可以连接到编码器、光电管或 PLC 等设备。

![](/public/upload/obtain_point_cloud/1.png)

安装成功后，传感器有指示灯显示

蓝色指示灯：通电成功

黄色指示灯：激光发射器激活

工作范围：

+ CD：270mm
+ MR：550mm
+ FOV：145-425mm

<img src="/public/upload/obtain_point_cloud/2.png" style="zoom:50%;" />

### 2. 软件配置

#### 2.1 设备端网络设置

通电后，将系统防火墙关闭，并将连接传感器得网卡 IP 地址修改成跟传感器同一网段。 传感器默认 IP 地址为 192.168.1.10。

<img src="/public/upload/obtain_point_cloud/3.png" style="zoom:50%;" />

#### 2.2 打开传感器界面

IP 地址修改完成后，浏览器地址栏输入传感器 IP 地址后即可进入，推荐使用谷歌浏览器，如进入到以下界面表示连接成功。

<img src="/public/upload/obtain_point_cloud/4.png" style="zoom:50%;" />

#### 2.3 软件调试

软件操作界面可参考[Gocator 界面概述及固件更新.pdf](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/obtain_point_cloud/Gocator%20%E7%95%8C%E9%9D%A2%E6%A6%82%E8%BF%B0%E5%8F%8A%E5%9B%BA%E4%BB%B6%E6%9B%B4%E6%96%B0.pdf)

##### 2.3.1 选取有效区域

放置被测物，时间模式开启传感器，调整合适曝光，视野中可见清晰轮廓线，前后移动平台（被测物）Y 方向并观察视图区轮廓，保证整个被测物都在视野中，如部分区域超出视野，需继续调整传感器与被测物之间距离直至实现。选择有效区域。 扫描界面→传感器→有效区域→选择，可出现黄色选择框，可以自由调节也可输入区域位置和大小，选择合适的有效区域并保证被测物所有位置都在有效区域内，设置完成后点击保存。

<img src="/public/upload/obtain_point_cloud/6.png" style="zoom:50%;" />

##### 2.3.2 点云的形成

点云其实是很多表面轮廓沿着垂直于轮廓线的方向（点云的 Y 方向）排列而成的立体图，如下图，要获取这样的图像，需要被测物和传感器一边做相对运动一边采集轮廓，这样传感器把采集到的所有不同位置的表面轮廓按照 Y 方向依次排列，也就形成了点云，我们通常把这个过程称为点云的扫描。

<img src="/public/upload/obtain_point_cloud/5.png" style="zoom:50%;" />

#### 2.4 点云获取

##### 2.4.1 校准

三维空间的点云是由二维点云轮廓切片按相等间距排列，此间距需要进行校准计算。

间距可以通过相对运动的速度与时间来计算。

使用圆盘校准的设置如下，使用自动时间来触发生成轮廓切片



<img src="/public/upload/obtain_point_cloud/7.png" style="zoom:50%;" />

<img src="/public/upload/obtain_point_cloud/8.png" style="zoom:50%;" />

<img src="/public/upload/obtain_point_cloud/9.png" style="zoom:50%;" />

##### 2.4.2 产品扫描

开启传送带与激光扫描开关后，将圆盘放置在传送带上，软件可自动完成校准。

完成间距校准后，开启激光扫描与传送带，将待测产品经过传送带时可以自动完成扫描与数据的临时缓存。

所有产品扫描完成后，关闭激光扫描，打开数据回放，将扫描的缓存点云数据保存为csv格式。

#####  2.4.3 csv转换为png

csv格式文件可以通过格式转换文件转换为深度图进行yolov8模型训练。

##### 2.4.4 csv转化为bin

```python
import csv
import struct
import os
def readpoint(f):
# 读取CSV文件
    xlist = []
    points = []
    with open(f, 'r') as file:
        csv_reader = csv.reader(file)
        y_row_index = None

        for row in csv_reader:
            if "Y\X" in row:
                # 找到包含"Y\X"的行，确定x轴和y轴的值所在的列
                y_row_index = csv_reader.line_num
                xlist = row[1:]
                break

        if y_row_index is not None:
            for row in csv_reader:
                while row[0] != 'End':
                    y_value = float(row[0])  # 第一个数据为y坐标
                    for z_str, x in zip(row[1:len(xlist)+1], xlist):
                        # print(z_str, x)
                        if z_str != '':
                            points.append([float(x), y_value, float(z_str)])
                    break
                else:
                    break

    return points

def csvtobin(input_folder, output_folder):
    try:
        # 创建新文件夹
        os.mkdir(output_folder)
        print(f"文件夹 '{output_folder}' 已创建成功。")

        # 获取输入文件夹中的所有文件
        files = os.listdir(input_folder)

        # 遍历文件夹中的文件
        for file in files:
            if file.endswith(".csv"):
                # 构建源文件路径和目标文件路径
                bin_file = file.split('.')[0] + '.bin'
                source_path = os.path.join(input_folder, file)
                target_path = os.path.join(output_folder, bin_file)

                points = readpoint(source_path)

                with open(target_path, 'wb') as file:
                    for x, y, z in points:
                        # 使用struct.pack将浮点数转换为二进制数据，并按需要的格式进行打包
                        binary_data = struct.pack('fff', x, y, z)
                        file.write(binary_data)
                # 将CSV文件复制到新文件夹

                print(f"已转换文件 '{file}' 到文件夹 '{output_folder}'。")

        print("所有CSV文件已成功复制到新文件夹。")
    except FileExistsError:
        print(f"文件夹 '{output_folder}' 已经存在。")
        # 获取输入文件夹中的所有文件
        files = os.listdir(input_folder)

        # 遍历文件夹中的文件
        for file in files:
            if file.endswith(".csv"):
                # 构建源文件路径和目标文件路径
                bin_file = file.split('.')[0] + '.bin'
                source_path = os.path.join(input_folder, file)
                target_path = os.path.join(output_folder, bin_file)

                points = readpoint(source_path)

                with open(target_path, 'wb') as file:
                    for x, y, z in points:
                        # 使用struct.pack将浮点数转换为二进制数据，并按需要的格式进行打包
                        binary_data = struct.pack('fff', x, y, z)
                        file.write(binary_data)
                # 将CSV文件复制到新文件夹

                print(f"已转换文件 '{file}' 到文件夹 '{output_folder}'。")

        print("所有CSV文件已成功复制到新文件夹。")
    except Exception as e:
        print(f"发生错误：{e}")
    return


# 指定输入文件夹和新建文件夹的名称
input_folder = "package_depth"  # 你的CSV文件所在的文件夹
output_folder = "package_depth_bin"  # 新建的文件夹名称

# 调用函数来处理CSV文件
csvtobin(input_folder, output_folder)


```

