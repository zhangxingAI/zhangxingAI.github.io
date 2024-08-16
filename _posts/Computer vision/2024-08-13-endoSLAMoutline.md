---

layout: post
title: endo-SLAM outline
category: 计算机视觉
tags: endo slam
keywords: end
typora-root-url: ../..

---


* TOC
{:toc}
## Endo-Slam outline

### Relative work

![mermaid-diagram-2024-08-15-184011](/public/upload/endo/endo10.png)

<img src="/public/upload/endo/endo11.png" alt="mermaid-diagram-2024-08-15-184011" style="zoom:50%;" />

<div style="text-align: center;">
    <img src="/public/upload/endo/endo7.png" width="300"/>
    <img src="/public/upload/endo/endo8.png" width="200"/>
</div>

<div style="text-align: center;">
    激光测距&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;三角测距
</div>

<div style="text-align: center;">
    <img src="/public/upload/endo/endo9.gif" width="600"/>
</div>

<div style="text-align: center;">
    结构光三维重建
</div>



### 数据集

主流测试公开数据集 [C3VD](https://durrlab.github.io/C3VD/)和[SimCol](https://github.com/anitarau/simcol)

### 临床应用场景

1. 消化道与病灶三维可视化
   - 实时重建
   - 可无级切换三维模型的观察视角
1. 病灶范围计算

### 难点

1. 镜面反射容易发生过曝光和欠曝光，比如反光表面
2. 半透明物体，如积液区域，将影响特征提取与特征匹配
3. 表面均色，将无法提取特征点
4. 高度非线性的器官变形
5. 场景杂波，如气泡、流体、血液
6. 定位漂移现象随重建序列变多而越明显
7. 图像质量：镜头移动过快造成二维图像有拖影
8. 现有标注数据均通过模型实验获取或计算机建模获取，与临床内镜视频数据有较大差距

### 论文

#### 重要论文与算法

- [SimCol3D](https://arxiv.org/html/2307.11261v2)

- [EndoGSLAM](https://arxiv.org/html/2403.15124v1)

- [ORB-SLAM3](https://arxiv.org/pdf/2007.11898) 【[开源代码](https://github.com/UZ-SLAMLab/ORB_SLAM3)】

- [MonoLoT](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10587075)

- [OneSLAM](https://link.springer.com/article/10.1007/s11548-024-03171-6)  FPS 2-4

-  [Endo-Depth-and-Motion](https://arxiv.org/pdf/2103.16525) 【[代码](https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion?tab=readme-ov-file)】补充资料：[Depth from Motion](https://blog.csdn.net/qq_39967751/article/details/126290811)

- [Artificial intelligence and automation in endoscopy and surgery](https://www.nature.com/articles/s41575-022-00701-y)

  - Endoscopic mapping and navigation p174

  

#### 特征提取

- **2021**  [Feature Descriptor Learning Based on Sparse Feature Matching](https://dl.acm.org/doi/10.1145/3511176.3511187)
  - 有效地从内窥镜图像中提取特征描述，同时获得更密集、更精确的匹配点
- **2023**   [Adaptive feature extraction method for capsule endoscopy images](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10710972/) 
  - 提升特征点匹配性能
- **2023**  [NORMAL RECONSTRUCTION FROM SPECULARITY IN THE ENDOSCOPIC SETTING](https://arxiv.org/pdf/2211.05642) 
  - 认为高亮区域为标准圆，其法线方向与椭圆投影有联系，以此解决高亮区域的三维重建

#### SLAM算法改进

- **2021**  [Effectiveness of synthetic data generation for capsule endoscopy](https://manage.effectpublishing.com/uploads/articles/635522269.pdf)
  - EndoSLAM model的应用与验证
- **2021**  [ORB-SLAM3](https://arxiv.org/pdf/2007.11898) 【[开源代码](https://github.com/UZ-SLAMLab/ORB_SLAM3)】 基于特征的视觉-惯性 SLAM 系统，该系统完全依赖于最大后验（MAP）估计，甚至在 IMU 初始化阶段也是如此。因此，该系统可在室内和室外、小型和大型环境中实时稳健运行，其精确度是以往方法的两到十倍。鲁棒性高。重建速度～200ms
- **2023**  [Bimodal Camera Pose Prediction for Endoscopy](https://arxiv.org/pdf/2204.04968) 
  - 使用虚拟模型创建结肠镜检查中相机位姿估计的数据集 SimCol，以及一种明确学习双峰分布来预测相机位姿的新方法
- **2023**  [OneSLAM](https://link.springer.com/article/10.1007/s11548-024-03171-6)  
  
  - 基于TAP模型，可跟踪多个帧的稀疏对应关系，并运行局部捆绑调整，以共同优化摄像机位姿和肠道的稀疏三维重建。改进了endo-SLAM，随序列变长漂移量变大，FPS 2-4。补充资料：[TAPIR: Tracking Any Point with per-frame Initialization and temporal ](https://deepmind-tapir.github.io/)  2023
- **2023**  [Endomapper dataset of complete calibrated endoscopy procedures](https://www.nature.com/articles/s41597-023-02564-7)
  - Endomapper 数据集，首个包含内窥镜校准和原始校准视频的内窥镜数据集。VSLAM 的目标是在内窥镜插入过程中实时绘制三维地图。人体表面的纹理较差，并且由于液体而产生大量反射。场景几何形状普遍存在变形。视频中既有对感兴趣区域的慢速观察，也有内窥镜镜头的快速运动和长时间遮挡。
- **2023**   [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/pdf/2308.04079)
  - 3D 高斯增强渲染真实性 >6min

- **2024**  [EndoGSLAM: Real-Time Dense Reconstruction and Tracking in Endoscopic Surgeries using Gaussian Splatting](https://arxiv.org/html/2403.15124v1)

  - 基于RGBD图像，提升重建速度与质量，速度100fps

- **2024**  [D EPTH A NYTHING IN M EDICAL I MAGES : A C OMPARATIVE S TUDY](https://arxiv.org/pdf/2401.16600) 

  - 通用大模型估计深度

  ![image-20240814154431182](/public/upload/endo/endo3.png)



#### 自监督网络

- **2020**  [EndoSLAM dataset and an unsupervised monocular visual odometry and depth estimation approach for endoscopic](https://arxiv.org/pdf/2006.16670)【[代码](https://github.com/CapsuleEndoscope/EndoSLAM)】

  - 内窥镜 SLAM 数据集，该数据集由六个猪器官的三维点云数据、胶囊和标准内窥镜记录、和计算机断层扫描（CT）扫描地面实况组成。从八个活体猪胃肠道器官和一个硅胶结肠模型中收集数据。总共提供了 35 个子数据集，其中体内部分为 6D 姿态Ground Truth：其中 18 个子数据集用于结肠，12 个子数据集用于胃，5 个子数据集用于小肠，而其中 4 个子数据集包含息肉。无监督的单目深度和姿态估计方法，它将残差网络与空间注意力模块相结合，以指示网络关注可区分的高纹理组织区域。

  ![image-20240814150736167](/public/upload/endo/endo2.png)

- **2021**  [Endo-Depth-and-Motion: Reconstruction and Tracking in Endoscopic Videos using Depth Networks and Photometric Constraints](https://arxiv.org/pdf/2103.16525)  【[代码](https://github.com/UZ-SLAMLab/Endo-Depth-and-Motion?tab=readme-ov-file)】补充资料：[Depth from Motion](https://blog.csdn.net/qq_39967751/article/details/126290811)

  - 利用自监督深度网络生成伪 RGBD 帧，然后跟踪相机姿态。

  The interest for applying SfM/SLAM in intracorporeal sequences has risen following the advances of the field, but encounters the challenges mentioned before. Early monocular approaches were based on the Extended Kalman Filter [36], [37]; and more recent ones on non-linear optimization for tracking and mapping [6] and map densification using variational approaches [38] or multi-view stereo [39]. These methods were strongly based on the rigidity assumption. MIS-SLAM [40], [41] was the first bringing deformable SLAM to intracorporeal images. It uses a canonical shape, as DynamicFusion [42], integrating stereo observations in a Truncated Signed Distance Function (TSDF) [43] with a deformation model. It uses the rigid tracking of ORB-SLAM2 [2] to estimate the camera pose between keyframes. DefSLAM [44] was the first monocular SLAM fully addressing deformations in monocular endoscopies. SD-DefSLAM [45] improves over it incorporating an illumination-invariant Lukas-Kanade tracker, relocalization and tool segmentation. Both of them use at their core an isometric NRSfM (IsoNRSfM) [10] over a sliding window and a robust deformation tracking inspired in [46]. Although IsoNRSfM models intracorporeal deformations, it assumes that the scene is a continuous surface, which does not hold for many in-body scenes. In addition, even in [45], feature correspondence keeps being a challenge. As another drawback, deformable tracking is computationally demanding. Compared to them, our Endo-Depth can be a fair substitute of IsoNRSfM for deformable SLAM. And, under the assumption of slow deformations, our high-keyframe-rate odometry allows Endo-Depth-and-Motion to achieve long tracks in both rigid and deformable in-body sequences.

  ![example input output gif](/public/upload/endo/endo6.gif)

- **2021**  [Self-Supervised Monocular Depth and Ego-Motion Estimation in Endoscopy: Appearance Flow to the Rescue](https://arxiv.org/pdf/2112.08122)  

  - 自监督单目深度估计框架 腹腔镜

- **2023**  [Self-supervised monocular depth estimation for gastrointestinal endoscopy](https://www.sciencedirect.com/science/article/pii/S0169260723002845)  

  - 一种自监督神经网络框架+双注意力机制，并行预测分支获得的深度信息和姿态信息重建图像，重建后的图像作为自监督信号指导网络模型训练。

- **2024**  [MonoLoT: Self-Supervised Monocular Depth Estimation in Low-Texture Scenes for Automatic Robotic Endoscopy](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10587075)  【[代码](https://github.com/howardchina/MonoLoT)】

  - 自监督单目深度估计框架 

    

#### 临床实验

- **2019**  [Comparison of 3D endoscopy and conventional 2D endoscopy in gastric endoscopic submucosal dissection: an ex vivo animal study](https://link.springer.com/article/10.1007/s00464-019-06726-w)  

  - 奥林巴斯预研的胃黏膜模型模拟实验，3D的效果通过佩戴3D眼镜来实现

  <img src="/public/upload/endo/endo4.png" alt="image-20240814160727605" style="zoom: 25%;" />

- **2021**   [Three-dimensional visualization improves the endoscopic diagnosis of superficial gastric neoplasia](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8161972/)  

  - 奥林巴斯3D内窥镜产品的临床应用，识别肿瘤边缘

  <img src="/public/upload/endo/endo5.png" alt="image-20240815095302650" style="zoom: 33%;" />

- **2022**  [Magnified endoscopy with texture and color enhanced imaging with indigo carmine for superficial nonampullary duodenal tumor:a pilot study](https://www.nature.com/articles/s41598-022-14476-4)   

  - 奥林巴斯实验探讨纹理和颜色增强成像（TXI）与放大内镜（ME）在浅表非髓质十二指肠上皮肿瘤（SNADET）术前诊断中的实用性。

  

