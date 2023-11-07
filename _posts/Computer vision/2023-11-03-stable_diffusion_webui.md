---

layout: post
title: Stable Diffusion
category: 计算机视觉
tags: stable_diffusion
keywords: stable_diffusion
typora-root-url: ../..

---



* TOC
{:toc}


## **0 参考资料**：

[深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识](https://zhuanlan.zhihu.com/p/643420260)

 [How to run SDXL v1.0 with AUTOMATIC1111](https://aituts.com/sdxl/)

[Stable Diffusion XL 1.0 model](https://stable-diffusion-art.com/sdxl-model/)

[超全面的Stable Diffusion学习指南：模型篇](https://stable-diffusion-art.com/sdxl-model/)

[从零开始训练专属 LoRA 模型](https://www.uisdc.com/lora-model)



## **1 Stable Diffusion XL算法原理**



### 1.1 Stable Diffusion XL核心基础内容
*Stable Diffusion XL是Stable Diffusion的优化版本。相当于：yolov8是 yolo的优化版本*

与Stable DiffusionV1-v2相比，Stable Diffusion XL主要做了如下的优化：

1. 对Stable Diffusion原先的U-Net，VAE，CLIP Text Encoder三大件都做了改进。
2. 增加一个单独的基于Latent的Refiner模型，来提升图像的精细化程度。
3. 设计了很多训练Tricks，包括图像尺寸条件化策略，图像裁剪参数条件化以及多尺度训练等。
4. **先发布Stable Diffusion XL 0.9测试版本，基于用户使用体验和生成图片的情况，针对性增加数据集和使用RLHF技术优化迭代推出Stable Diffusion XL 1.0正式版**。

### 1.2 SDXL整体架构

Stable Diffusion XL是一个**二阶段的级联扩散模型**，包括Base模型和Refiner模型。其中Base模型的主要工作和Stable Diffusion一致，具备文生图，图生图，图像inpainting等能力。在Base模型之后，级联了Refiner模型，对Base模型生成的图像Latent特征进行精细化，**其本质上是在做图生图的工作**。Base模型用于生成有噪声的latent，这些latent在最后的去噪步骤中使用专门的Refiner模型进行处理。Base模型可以单独使用，但Refiner模型只能依附base模式使用提高图像的清晰度和质量。

**Base模型由U-Net，VAE，CLIP Text Encoder（两个）三个模块组成**，在FP16精度下Base模型大小6.94G（FP32：13.88G），其中U-Net大小5.14G，VAE模型大小167M以及两个CLIP Text Encoder一大一小分别是1.39G和246M。

**Refiner模型同样由U-Net，VAE，CLIP Text Encoder（一个）三个模块组成**，在FP16精度下Refiner模型大小6.08G，其中U-Net大小4.52G，VAE模型大小167M（与Base模型共用）以及CLIP Text Encoder模型大小1.39G（与Base模型共用）。

<img src="/public/upload/SDXL/1.png" alt="SDXL model pipeline" style="zoom:50%;" />

![](/public/upload/SDXL/2.png)

1. **GSC模块：**Stable Diffusion Base XL U-Net中的最小组件之一，由GroupNorm+SiLU+Conv三者组成。
2. **DownSample模块：**Stable Diffusion Base XL U-Net中的下采样组件，**使用了Conv（kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)）进行采下采样**。
3. **UpSample模块：**Stable Diffusion Base XL U-Net中的上采样组件，由**插值算法（nearest）**+Conv组成。
4. **ResNetBlock模块：**借鉴ResNet模型的“残差结构”，**让网络能够构建的更深的同时，将Time Embedding信息嵌入模型**。
5. **CrossAttention模块：**将文本的语义信息与图像的语义信息进行Attention机制，增强输入文本Prompt对生成图片的控制。
6. **SelfAttention模块：**SelfAttention模块的整体结构与CrossAttention模块相同，这是输入全部都是图像信息，不再输入文本信息。
7. **FeedForward模块：**Attention机制中的经典模块，由GeGlU+Dropout+Linear组成。
8. **BasicTransformer Block模块：**由LayerNorm+SelfAttention+CrossAttention+FeedForward组成，是多重Attention机制的级联，并且每个Attention机制都是一个“残差结构”。**通过加深网络和多Attention机制，大幅增强模型的学习能力与图文的匹配能力**。
9. **SDXL_Spatial Transformer_X模块：**由GroupNorm+Linear+**X个BasicTransformer Block**+Linear构成，同时ResNet模型的“残差结构”依旧没有缺席。
10. **SDXL_DownBlock模块：**由ResNetBlock+ResNetBlock+DownSample组成。
11. **SDXL_UpBlock_X模块：**由X个ResNetBlock模块组成。
12. **CrossAttnDownBlock_X_K模块：**是Stable Diffusion XL Base U-Net中Encoder部分的主要模块，由K个**（ResNetBlock模块+SDXL_Spatial Transformer_X模块）**+DownSample模块组成。
13. **CrossAttnUpBlock_X_K模块：**是Stable Diffusion XL Base U-Net中Decoder部分的主要模块，由K个**（ResNetBlock模块+SDXL_Spatial Transformer_X模块）**+UpSample模块组成。
14. **CrossAttnMidBlock模块：**是Stable Diffusion XL Base U-Net中Encoder和ecoder连接的部分，由ResNetBlock+**SDXL_Spatial Transformer_10**+ResNetBlock组成。

可以看到，其中增加的**SDXL_Spatial Transformer_X模块（主要包含Self Attention + Cross Attention + FeedForward）**数量占新增参数量的主要部分，Rocky在上表中已经用红色框圈出。U-Net的Encoder和Decoder结构也从原来的4stage改成3stage（[1,1,1,1] -> [0,2,10]），说明SDXL只使用两次下采样和上采样，而之前的SD系列模型都是三次下采样和上采样。并且比起Stable DiffusionV1/2，Stable Diffusion XL在第一个stage中不再使用Spatial Transformer Blocks，而在第二和第三个stage中大量增加了Spatial Transformer Blocks（分别是2和10），**那么这样设计有什么好处呢？**

首先，**在第一个stage中不使用SDXL_Spatial Transformer_X模块，可以明显减少显存占用和计算量。**然后在第二和第三个stage这两个维度较小的feature map上使用数量较多的SDXL_Spatial Transformer_X模块，能**在大幅提升模型整体性能（学习能力和表达能力）的同时，优化了计算成本**。整个新的SDXL Base U-Net设计思想也让SDXL的Base出图分辨率提升至1024x1024。在参数保持一致的情况下，**Stable Diffusion XL生成图片的耗时只比Stable Diffusion多了20%-30%之间，这个拥有2.6B参数量的模型已经足够伟大**。

在SDXL U-Net的Encoder结构中，包含了两个CrossAttnDownBlock结构和一个SDXL_DownBlock结构；在Decoder结构中，包含了两个CrossAttnUpBlock结构和一个SDXL_UpBlock结构；与此同时，Encoder和Decoder中间存在Skip Connection，进行信息的传递与融合。

**BasicTransformer Block模块是整个框架的基石，由SelfAttention，CrossAttention和FeedForward三个组件构成，并且使用了循环残差模式，让SDXL Base U-Net不仅可以设计的更深，同时也具备更强的文本特征和图像体征的学习能力**。

**Stable Diffusion XL中的Text Condition信息由两个Text Encoder提供（OpenCLIP ViT-bigG和OpenAI CLIP ViT-L）**，通过Cross Attention组件嵌入，作为K Matrix和V Matrix。与此同时，图片的Latent Feature作为Q Matrix。

### 1.3 VAE模型

VAE模型（变分自编码器，Variational Auto-Encoder）是一个经典的生成式模型。在传统深度学习时代，GAN的风头完全盖过了VAE，但VAE简洁稳定的Encoder-Decoder架构，**以及能够高效提取数据Latent特征的关键能力**，让其跨过了周期，在AIGC时代重新繁荣。

Stable Diffusion XL依旧是**基于Latent**的扩散模型，**所以VAE的Encoder和Decoder结构依旧是Stable Diffusion XL提取图像Latent特征和图像像素级重建的关键一招**。

当输入是图片时，Stable Diffusion XL和Stable Diffusion一样，首先会使用VAE的**Encoder结构将输入图像转换为Latent特征**，然后U-Net不断对Latent特征进行优化，最后使用VAE的**Decoder结构将Latent特征重建出像素级图像**。除了提取Latent特征和图像的像素级重建外，**VAE还可以改进生成图像中的高频细节，小物体特征和整体图像色彩**。

当Stable Diffusion XL的输入是文字时，这时我们不需要VAE的Encoder结构，只需要Decoder进行图像重建。VAE的灵活运用，让Stable Diffusion系列增添了几分优雅。

**Stable Diffusion XL使用了和之前Stable Diffusion系列一样的VAE结构**，但在训练中选择了**更大的Batch-Size（256 vs 9）**，并且对模型进行指数滑动平均操作（**EMA**，exponential moving average），EMA对模型的参数做平均，从而提高性能并增加模型鲁棒性。

![img](/public/upload/SDXL/3.png)

在损失函数方面，使用了久经考验的**生成领域“交叉熵”—感知损失（perceptual loss）**以及回归损失来约束VAE的训练过程。

与此同时，VAE的**缩放系数**也产生了变化。VAE在将Latent特征送入U-Net之前，需要对Latent特征进行缩放让其标准差尽量为1，之前的Stable Diffusion系列采用的**缩放系数为0.18215，**由于Stable Diffusion XL的VAE进行了全面的重训练，所以**缩放系数重新设置为0.13025**。

注意：由于缩放系数的改变，Stable Diffusion XL VAE模型与之前的Stable Diffusion系列并不兼容。

### 1.4 CLIP Text Encoder模型

**CLIP模型主要包含Text Encoder和Image Encoder两个模块**，在Stable Diffusion XL中，和之前的Stable Diffusion系列一样，**只使用Text Encoder模块从文本信息中提取Text Embeddings**。

**不过Stable Diffusion XL与之前的系列相比，使用了两个CLIP Text Encoder，分别是OpenCLIP ViT-bigG（1.39G）和OpenAI CLIP ViT-L（246M），从而大大增强了Stable Diffusion XL对文本的提取和理解能力。**

其中OpenCLIP ViT-bigG是一个只由Transformer模块组成的模型，一共有32个CLIPEncoder模块，是一个强力的特征提取模型。

OpenAI CLIP ViT-L同样是一个只由Transformer模块组成的模型，一共有12个CLIPEncoder模块。

OpenCLIP ViT-bigG的优势在于模型结构更深，特征维度更大，特征提取能力更强，但是其两者的基本CLIPEncoder模块是一样的。

**与传统深度学习中的模型融合类似**，Stable Diffusion XL分别提取两个Text Encoder的倒数第二层特征，并进行concat操作作为文本条件（Text Conditioning）。其中OpenCLIP ViT-bigG的特征维度为77x1280，而CLIP ViT-L的特征维度是77x768，所以输入总的特征维度是77x2048（77是最大的token数），再通过Cross Attention模块将文本信息传入Stable Diffusion XL的训练过程与推理过程中。

### 1.5 Refiner模型

DeepFloyd和StabilityAI联合开发的DeepFloyd IF**是一种基于像素的文本到图像三重级联扩散模型**，大大提升了扩散模型的图像生成能力。

这次，Stable Diffusion XL终于也开始使用级联策略，在U-Net（Base）之后，级联Refiner模型，进一步提升生成图像的细节特征与整体质量。

**通过级联模型提升生成图片的质量，可以说这是AIGC时代里的模型融合。和传统深度学习时代的多模型融合策略一样，不管是学术界，工业界还是竞赛界，都是“行业核武”般的存在。**

![img](/public/upload/SDXL/4.png)

由于已经有U-Net（Base）模型生成了图像的Latent特征，所以**Refiner模型的主要工作是在Latent特征进行小噪声去除和细节质量提升**。

**Refiner模型和Base模型一样是基于Latent的扩散模型**，也采用了Encoder-Decoder结构，和U-Net兼容同一个VAE模型，不过Refiner模型的Text Encoder只使用了OpenCLIP ViT-bigG。

**下图是Stable Diffusion XL Refiner模型的完整结构图**

![img](/public/upload/SDXL/5.webp)

在Stable Diffusion XL推理阶段，输入一个prompt，通过VAE和U-Net（Base）模型生成Latent特征，接着给这个Latent特征加一定的噪音，在此基础上，再使用Refiner模型进行去噪，以提升图像的整体质量与局部细节。

<img src="/public/upload/SDXL/6.png" alt="img" style="zoom: 50%;" />

可以看到，**Refiner模型主要做了图像生成图像的工作**，其具备很强的**迁移兼容能力**，可以作为Stable Diffusion，GAN，VAE等生成式模型的级联组件，不管是对学术界，工业界还是竞赛界，无疑都是一个无疑都是一个巨大利好。

<img src="/public/upload/SDXL/7.png" alt="img" style="zoom: 50%;" />

只使用U-Net（Base）模型，Stable Diffusion XL模型的效果已经大幅超过SD1.5和SD2.1，当增加Refiner模型之后，Stable Diffusion XL达到了更加优秀的图像生成效果。

## **2 常用的Stable Diffusion模型**

### 2.1 checkpoint模型

checkpoint模型是真正意义上的Stable Diffusion模型，它们包含生成图像所需的一切（TextEncoder， U-net， VAE），不需要额外的文件。但是它们体积很大，通常为2G-7 G。官方的Stable Diffusion v1-5 版本模型的训练使用了 256 个 40G 的 A100 GPU，合计耗时 15 万个 GPU 小时（约 17 年），总成本达到了 60 万美元。除此之外，为了验证模型的出图效果，伴随着上万名测试人员每天 170 万张的出图测试。Stable Diffusion 作为专注于图像生成领域的大模型，它的目的并不是直接进行绘图，而是通过学习海量的图像数据来做预训练，提升模型整体的基础知识水平，这样就能以强大的通用性和实用性状态完成后续下游任务的应用。Stable Diffusion 官方模型的真正价值在于降低了模型训练的门槛，因为在现有大模型基础上训练新模型的成本要低得多。对特定图片生成任务来说，只需在官方模型基础上加上少量的文本图像数据，并配合微调模型的训练方法，就能得到应用于特定领域的定制模型。一方面训练成本大大降低，只需在本地用一张民用级显卡训练几小时就能获得稳定出图的定制化模型，另一方面，针对特定方向训练模型的理解和绘图能力更强，实际的出图效果反而有了极大的提升。

Checkpoint 模型的常见训练方法叫 Dreambooth，该技术原本由谷歌团队基于自家的 Imagen 模型开发，后来经过适配被引入 Stable Diffusion 模型中，并逐渐被广泛应用。![12](/public/upload/SDXL/12.png)

### 2.2 Textual lnversion

Textual lnversion（又叫Embedding）是定义新关键字以生成新人物或图片风格的小文件。Stable Diffusion 模型包含文本编码器、扩散模型和图像编码器 3 个部分，其中文本编码器 TextEncoder 的作用是将提示词转换成电脑可以识别的文本向量，而 Embedding 模型的原理就是通过训练将包含特定风格特征的信息映射在其中，这样后续在输入对应关键词时，模型就会自动启用这部分文本向量来进行绘制。它们很小，通常为10-100 KB。必须将它们与checkpoint模型一起使用。（2023-11-06时间段中实际应用中效果不佳，不推荐使用）![截屏2023-11-07 11.28.45](./public/upload/SDXL/13.png)

### 2.3 LoRA 模型

LoRA 模型是用于修改图片风格的checkpoint模型的小补丁文件。它们通常为10-200 MB。必须与checkpoint模型一起使用。LoRA 是 Low-Rank Adaptation Models 的缩写，意思是低秩适应模型。LoRA 原本并非用于 AI 绘画领域，它是微软的研究人员为了解决大语言模型微调而开发的一项技术，因此像 GPT3.5 包含了 1750 亿量级的参数，如果每次训练都全部微调一遍体量太大，而有了 lora 就可以将训练参数插入到模型的神经网络中去，而不用全面微调。通过这样即插即用又不破坏原有模型的方法，可以极大的降低模型的训练参数，模型的训练效率也会被显著提升。![](/public/upload/SDXL/15.png)

相较于 Dreambooth 全面微调模型的方法，LoRA 的训练参数可以减少上千倍，对硬件性能的要求也会急剧下降，如果说 Embeddings 像一张标注的便利贴，那 LoRA 就像是额外收录的夹页，在这个夹页中记录了更全面图片特征信息。

由于需要微调的参数量大大降低，LoRA 模型的文件大小通常在几百 MB，比 Embeddings 丰富了许多，但又没有 ckpt 那么臃肿。模型体积小、训练难度低、控图效果好，可以说是目前最热门的模型之一。

### 2.4 Hypernetwork 模型

Hypernetwork是添加到checkpoint模型中的附加网络模块。它们通常为5-300 MB。必须与checkpoint模型一起使用。它的原理是在扩散模型之外新建一个神经网络来调整模型参数，而这个神经网络也被称为超网络。因为 Hypernetwork 训练过程中同样没有对原模型进行全面微调，因此模型尺寸通常也在几十到几百 MB 不等。它的实际效果，我们可以将其简单理解为低配版的 LoRA，虽然超网络这名字听起来很厉害，但其实这款模型如今的风评并不出众，在国内已逐渐被 lora 所取代。因为它的训练难度很大且应用范围较窄，目前大多用于控制图像画风。所以除非是有特定的画风要求，否则还是建议优先选择 LoRA 模型来使用。![](/public/upload/SDXL/14.png)

## **3 实践**

### 3.1 主流方法
目前可以在本地运行SDXL的主流方法为

- **[AUTOMATIC1111's Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)**最流行的 WebUI，拥有最多的功能和扩展，最容易学习。本说明书将使用此方法
- **[ComfyUI ](https://github.com/comfyanonymous/ComfyUI)**上手较难，node based interface，但生成速度非常快，比 AUTOMATIC1111 快 5-10 倍。允许使用两种不同的正向提示。

### 3.2 AUTOMATIC1111使用方法

#### 3.2.1  Install AUTOMATIC1111 on Linux

```cmd
# Create environment
conda create -n sd python=3.10.6
# Activate environment
conda activate sd
# Validate environment is selected
conda info --env
# Start local webserver
./webui.sh
# Wait for "Running on local URL:  http://127.0.0.1:7860" and open that URI.
```

#### 3.2.2  下载模型

下载base model 和 refiner model（每个～6GB）[huggingface网站](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

- 将base model 和 refiner model放置在./model/stable-diffusion-webui/中

- 运行

  ```cmd
  ./webui.sh --xformers
  ```

- 打开url链接http://127.0.0.1:7860，在以下的ui选择框中可以选择对应的model

base model作为训练或推理时的底模使用

<img src="/public/upload/SDXL/8.png" alt="img" style="zoom: 50%;" />

refiner model仅在推理时使用

<img src="/public/upload/SDXL/9.png" alt="Refiner model selector " style="zoom: 50%;" />

#### 3.2.2 调参生图

- **Resolution**: 1024 x 1024 

- **Tiling** 生成一个可以平铺的图像；

- **Highres. fix** 使用两个步骤的过程进行生成，以base model创建图像，然后在不改变构图的情况下以refine model改进其中的细节，选择该部分会有两个新的参数

- **Switch at**: 切换位置，该值控制在哪一步切换到refiner model。例如，在 0.5 时切换并使用 40 个步骤表示在前 20 个步骤中使用base model，在后 20 个步骤中使用refiner model。

  如果切换到 1，则只使用base model。

- **Sampling Steps**：经验最优值为**30 steps** and **switch at 0.6**。diffusion model 生成图片的迭代步数，每多一次迭代都会给 AI 更多的机会去比对 prompt 和 当前结果，去调整图片。更高的步数需要花费更多的计算时间，也相对更贵。但不一定意味着更好的结果。

<img src="/public/upload/SDXL/10.png" style="zoom:33%;" />

- **Clip skip**: 1 or 2
- **Sampling method** 扩散去噪算法的采样模式，会带来不一样的效果，ddim 和 pms(plms) 的结果差异会很大，很多人还会使用euler，具体没有系统测试。
- **Tiling** 生成一个可以平铺的图像；

  **Highres. fix** 使用两个步骤的过程进行生成，以较小的分辨率创建图像，然后在不改变构图的情况下改进其中的细节，选择该部分会有两个新的参数
- **CFG scale**: 7。分类器自由引导尺度——图像与提示符的一致程度——越低的值产生越有创意的结果；
- **Seed** 种子数，只要种子数一样，参数一致、模型一样图像就能重新被绘制；
-  **Scale latent** 在潜空间中对图像进行缩放。另一种方法是从潜在的表象中产生完整的图像，将其升级，然后将其移回潜在的空间。
- **Denoising strength**:  决定算法对图像内容的保留程度。在0处，什么都不会改变，而在1处，会得到一个不相关的图像。一般大于0.7出来的都是新效果，小于0.3基本就会原图缝缝补补；

<img src="/public/upload/SDXL/11.png" style="zoom:33%;" />

- 增加选择已训练的lora和hypernetwork模型可以实现更多自定义功能

  例如，把训练好的 LoRA 模型全部放入 LoRA 模型目录 stable-diffusion-webui/models/Lora。

  ![](/public/upload/SDXL/20.jpg)

#### tips：使用XYZ plot 脚本

- 在 Stable Diffusion WebUI 页面最底部的脚本栏中调用 XYZ plot 脚本，设置模型对比参数。

- **其中 X 轴类型和 Y 轴类型都选择「提示词搜索/替换」Prompt S/R。**

​	X 轴值输入：NUM,000001,000002,000003,000004,000005，对应模型序号

​	Y 轴值输入：STRENGTH,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1，对应模型权重值

<img src="/public/upload/SDXL/22.jpg" style="zoom: 33%;" />

- 把引入的 LoRA 模型提示词，改成变量模式，如： 改成 ，NUM 变量代表模型序号，STRENGTH 变量代表权重值。

![](/public/upload/SDXL/21.jpg)

- 通过对比生成结果，选出表现最佳的模型和权重值。

<img src="./public/upload/SDXL/23.jpg" style="zoom:50%;" />

### 3.3 训练自定义模型

#### 3.3.1 准备数据集

根据需要训练的标签来命名文件夹，例如选取的训练标签名称有：qzy, lsbr, flawless, flawed, fold, spot

文件存放的示意结构如下

```
weipin
├── qzy
│   ├── flawless
│   │   ├── xxx1.jpg
│   │   ├── xxx2.jpg
│   │   ├── xxx3.jpg
│   ├── flawed
│   │   ├── fold
│   │   │   ├── xxx1.jpg
│   │   │   ├── xxx2.jpg
│   │   ├── spot
│   │   │   ├── xxx1.jpg
│   │   │   ├── xxx2.jpg
├── lsbr
│   ├── flawless
│   │   ├── xxx1.jpg
│   │   ├── xxx2.jpg
│   │   ├── xxx3.jpg
│   ├── flawed
│   │   ├── fold
│   │   │   ├── xxx1.jpg
│   │   │   ├── xxx2.jpg
│   │   ├── spot
│   │   │   ├── xxx1.jpg
│   │   │   ├── xxx2.jpg

```

每个末端文件夹中的图片建议数量为5-10张，使用`filename.py`将所有jpg文件重命名为`母文件夹名_子文件夹名_数字编号`的形式，如`qzy_flawed_spot_数字编号`并将所有图片文件归到一个文件夹下

```
train_output
├── qzy_flawed_spot_01.jpg
├── qzy_flawed_spot_02.jpg
├── qzy_flawed_spot_03.jpg
├── lsbr_flawless_01.jpg
```

```python
####filename.py

import os
import shutil

def rename_and_copy_files(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        if files:  # 只处理当前文件夹中包含文件的情况
            folder_names = os.path.normpath(root).split(os.path.sep)  # 获取文件夹路径并拆分为名称列表
            new_name = "_".join(folder_names[1:])  # 使用下划线将文件夹名称拼接成新名称
            i = 1  # 用于追踪重命名文件的数字
            for file_name in files:
                file_extension = os.path.splitext(file_name)[1]  # 获取文件扩展名
                new_file_name = f"{new_name}_{i:02d}{file_extension}"  # 新文件名
                i += 1
                file_path = os.path.join(root, file_name)
                new_file_path = os.path.join(root, new_file_name)
                os.rename(file_path, new_file_path)

                # 构建目标文件的完整路径
                destination_file_path = os.path.join(destination_folder, new_file_name)

                # 复制文件到目标文件夹
                shutil.copy(new_file_path, destination_file_path)

source_folder = 'products'  # 请替换为您的源文件夹路径
destination_folder = 'zx_train_output'  # 请替换为您的目标文件夹路径

rename_and_copy_files(source_folder, destination_folder)

```



- 打开webui，使用图片剪裁功能将图片批量剪裁为512 x 512（hypernetwork训练）或1024 x 1024（lora训练）。根据blip处理规则自动重新命名图片文件，并生成与图片名一致的txt文件。

  <img src="/public/upload/SDXL/16.png" alt="img" style="zoom:50%;" />

- 通过提取文件名中的关键词，将自动生成的txt文件中的内容修改为将要训练的标签，用逗号隔开。使用`fixtxt.py`批量处理

```python
### fixtxt.py

import os

# 定义要搜索的文件夹路径
folder_path = 'zx_train'
add = 'black background'
# 遍历文件夹中的所有文件
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)

            # 提取文件名中的关键词以_作为分隔符
            keywords = os.path.splitext(file)[0].split('-')[-1].split('_')[:-1]

            keywords.append(add)

            # 将关键词以逗号分隔的形式写入原始文件
            with open(file_path, 'w') as f:
                f.write(', '.join(keywords))
```

经过以上捕捉，训练数据集准备完成

#### 3.3.2 训练hypernetwork模型

可直接在AUTOMATIC1111's Stable Diffusion WebUI中进行。

- 首先在Create hypernetwork中创建一个hypernetwork
- 然后在train中添加数据集开始训练

<img src="/public/upload/SDXL/18.png" style="zoom:33%;" />

数据集结构为：

```
weipin
├── xxx1.jpg
├── xxx1.txt
├── xxx2.jpg
├── xxx2.txt
```

#### 3.3.3 训练lora模型

lora模型的训练需要借助kohya_ss。

- 安装kohya_ss

  ```cmd
  # 1. clone repo
  git clone -b sdxl-dev https://github.com/bmaltais/kohya_ss.git
  
  # 2. install torch
  #cu117
  conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
  #cu118
  conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
  
  # 3.install requirements
  # install with file
  pip install -r requirements.txt
  # install without file
  pip install accelerate==0.19.0 albumentations==1.3.0 altair==4.2.2 dadaptation==3.1 diffusers[torch]==0.18.2 easygui==0.98.3 einops==0.6.0 fairscale==0.4.13 ftfy==6.1.1 gradio==3.36.1 huggingface-hub==0.15.1 lion-pytorch==0.0.6 lycoris_lora==1.8.0 invisible-watermark==0.2.0 open-clip-torch==2.20.0 opencv-python==4.7.0.68 prodigyopt==1.0 pytorch-lightning==1.9.0 rich==13.4.1 safetensors==0.3.1 timm==0.6.12 tk==0.1.0 toml==0.10.2 transformers==4.30.2 voluptuous==0.13.1 wandb==0.15.0 xformers==0.0.20 bitsandbytes==0.35.0 tensorboard==2.12.3 tensorflow==2.12.0
  # if you are using linux
  pip install xformers==0.0.20 bitsandbytes==0.35.0 tensorboard==2.12.3 tensorflow==2.12.0
  ```

- 使用koyha_ss

  - 在根目录下运行

    ```cmd
    ./gui.sh
    ```

  - 初始运行会从huggingface.co网站下载各种预训练模型，如果下载失败，运行过程中，大概率报错。因此需要手动下载以下文件夹并根据报错信息将加载模型的路径改为本地的文件夹路径。

    [bert-base-uncased](https://huggingface.co/bert-base-uncased)

    [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)

    [CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k)

  - 打开UI界面

  ![](/public/upload/SDXL/19.png)

  

  - 在LoRA界面中，选择weipin.json文件加载已经设置好的各项参数。[weipin.json](https://github.com/zhangxingAI/zhangxingAI.github.io/blob/main/public/upload/SDXL/weipin.json)

  - 设置训练数据集的路径并开始训练

  训练数据集结构为：

  ```
  weipin
  ├── image
  │   ├── 80_weipin
  │   │   ├── xxx1.jpg
  │   │   ├── xxx1.txt
  │   │   ├── xxx2.jpg
  │   │   ├── xxx2.txt
  ├── model
  ├── log
  ```

  其中80_weipin中的80代表训练图片80次，**注意：image文件夹的结构不能改变**。

  - 若image文件夹中有.ipynb_checkpoints隐藏文件，则会报错，使用以下代码删除.ipynb_checkpoints。

  ```cmd
  rm -rf .ipynb_checkpoints
  find . -name ".ipynb_checkpoints" -exec rm -rf {} \;  ## 这个在大文件运行后面就都删了 因为.ipynb_checkpoints是文件夹 需要加-rf循环地删除
  ```

- LoRA的训练参数
  1. 底模：填入底模文件夹地址 /content/Lora/sd_model/，刷新加载底模。
     resolution：训练分辨率，支持非正方形，但必须是 64 倍数。一般方图 1024x1024。
  2. batch_size：一次性送入训练模型的样本数，显存小推荐 1，24G取3-4，并行数量越大，训练速度越快。
  3. optimizer:  选择AdamW较多
  4. max_train_epoches：最大训练的 epoch 数，即模型会在整个训练数据集上循环训练的次数。如最大训练 epoch 为 10，那么训练过程中将会进行 10 次完整的训练集循环，一般可以设为 30。
  5. network_dim：线性 dim，代表模型大小，数值越大模型越精细，常用 128，如果设置为 128，则 LoRA 模型大小为 144M。
  6. network_alpha：线性 alpha，一般设置为比 Network Dim 小或者相同，通常将 network dim 设置为 128，network alpha 设置为 64。
  7. unet_lr：unet 学习率，5e-05
  8. text_encoder_lr：文本编码器的学习率，1e-04
  9. lr_scheduler：学习率调度器，用来控制模型学习率的变化方式，constant_with_warmup。
  10. lr_warmup_steps：升温步数，仅在学习率调度策略为“constant_with_warmup”时设置，用来控制模型在训练前逐渐增加学习率的步数，可以设为10%。



