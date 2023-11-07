---

layout: post
title: Stable Diffusion
category: 计算机视觉
tags: 
keywords:
typora-root-url: ../..

---

### Stable Diffusion

参考资料：

[深入浅出完整解析Stable Diffusion XL（SDXL）核心基础知识](https://zhuanlan.zhihu.com/p/643420260)（作者Rocky Ding）

 [How to run SDXL v1.0 with AUTOMATIC1111](https://aituts.com/sdxl/)（作者 Yubin）

[Stable Diffusion XL 1.0 model](https://stable-diffusion-art.com/sdxl-model/)（作者Andrew）

bash webui.sh

插件安装：

- 从网址安装，在/stable-diffusion-webui/extensions扩展文件夹中打开终端，使用git clone命令安装。比如以安装tagcomplete插件为例，输入：git clone
  https://github.com/DominikDoom/a1111-sd-webui-tagcomplete，按回车即可进行自动克隆安装。
- 从github上下载插件的zip包，解压缩后放入/stable-diffusion-webui/extensions扩展文件夹，重启webUI即可。

Stable Diffusion模型下载网站

[http://civitai.com](https://link.zhihu.com/?target=http%3A//civitai.com)和[http://huggingface.co](https://link.zhihu.com/?target=http%3A//huggingface.co)

具体模型类型又有checkpoint、Textual lnversion、Hypernetwork、Aesthetic Gradient、LoRA

LyCORIS、Controlnet、Poses、wildcards等

★checkpoint模型是真正意义上的Stable Diffusion模型，它们包含生成图像所需的一切，不需要额外的文件。但是它们体积很大，通常为2G-7 G。

目前比较流行和常见的checkpoint模型有Anythingv3、Anythingv4.5、AbyssOrangeMix3、counterfeitV2.5、PastalMix、CamelliaMix_2.5D、chilloutMix_Ni_fix、F222、openjourney等。这些checkpoint模型是从Stable Diffusion基本模型训练而来的，相当于基于原生安卓系统进行的二次开发。目前，大多数模型都是从 v1.4 或 v1.5 训练的。它们使用其他数据进行训练，以生成特定风格或对象的图像。

Anything、Waifu、novelai、Counterfeit是二次元漫画模型，比较适合生成动漫游戏图片

★Textual lnversion（又叫Embedding）是定义新关键字以生成新人物或图片风格的小文件。它们很小，通常为10-100 KB。必须将它们与checkpoint模型一起使用。

★LoRA 模型是用于修改图片风格的checkpoint模型的小补丁文件。它们通常为10-200 MB。必须与checkpoint模型一起使用。

★Hypernetwork是添加到checkpoint模型中的附加网络模块。它们通常为5-300 MB。必须与checkpoint模型一起使用。

★Aesthetic Gradient是一个功能，它将准备好的图像数据的方向添加到“Embedding”中，将输入的提示词转换为矢量表示并定向图像生成。

★LyCORIS：LyCORIS可以让LoRA学习更多的层，可以当做是升级的LoRA



UI界面

- txt2img --- 标准的文字生成图像；
- img2img --- 根据图像成文范本、结合文字生成图像；
- Extras --- 优化(清晰、扩展)图像；
- PNG Info --- 图像基本信息
- Checkpoint Merger --- 模型合并
- Textual inversion --- 训练模型对于某种图像风格
- Settings --- 默认参数修改

**Sampling Steps** diffusion model 生成图片的迭代步数，每多一次迭代都会给 AI 更多的机会去比对 prompt 和 当前结果，去调整图片。更高的步数需要花费更多的计算时间，也相对更贵。但不一定意味着更好的结果。当然迭代步数不足（少于 50）肯定会降低结果的图像质量；

**Sampling method** 扩散去噪算法的采样模式，会带来不一样的效果，ddim 和 pms(plms) 的结果差异会很大，很多人还会使用euler，具体没有系统测试；

**Width、Height** 图像长宽，可以通过send to extras 进行扩大，所以这里不建议设置太大[显存小的特别注意]；

**Restore faces** 优化面部，绘制面部图像特别注意；

**Tiling** 生成一个可以平铺的图像；

**Highres. fix** 使用两个步骤的过程进行生成，以较小的分辨率创建图像，然后在不改变构图的情况下改进其中的细节，选择该部分会有两个新的参数

 **Scale latent** 在潜空间中对图像进行缩放。另一种方法是从潜在的表象中产生完整的图像，将其升级，然后将其移回潜在的空间。Denoising strength 决定算法对图像内容的保留程度。在0处，什么都不会改变，而在1处，你会得到一个不相关的图像；

**Batch count、 Batch size** 都是生成几张图，前者计算时间长，后者需要显存大；

**CFG Scale** 分类器自由引导尺度——图像与提示符的一致程度——越低的值产生越有创意的结果；

**Seed** 种子数，只要种子数一样，参数一致、模型一样图像就能重新被绘制；

**Denoising strength** 与原图一致性的程度，一般大于0.7出来的都是新效果，小于0.3基本就会原图缝缝补补；





Stable Diffusion XL是Stable Diffusion的优化版本

相当于：yolov8是 yolo的优化版本

#### Stable Diffusion XL核心基础内容

与Stable DiffusionV1-v2相比，Stable Diffusion XL主要做了如下的优化：

1. 对Stable Diffusion原先的U-Net，VAE，CLIP Text Encoder三大件都做了改进。
2. 增加一个单独的基于Latent的Refiner模型，来提升图像的精细化程度。
3. 设计了很多训练Tricks，包括图像尺寸条件化策略，图像裁剪参数条件化以及多尺度训练等。
4. **先发布Stable Diffusion XL 0.9测试版本，基于用户使用体验和生成图片的情况，针对性增加数据集和使用RLHF技术优化迭代推出Stable Diffusion XL 1.0正式版**。

##### SDXL整体架构

Stable Diffusion XL是一个**二阶段的级联扩散模型**，包括Base模型和Refiner模型。其中Base模型的主要工作和Stable Diffusion一致，具备文生图，图生图，图像inpainting等能力。在Base模型之后，级联了Refiner模型，对Base模型生成的图像Latent特征进行精细化，**其本质上是在做图生图的工作**。Base模型用于生成有噪声的latent，这些latent在最后的去噪步骤中使用专门的Refiner模型进行处理。Base模型可以单独使用，但Refiner模型只能依附base模式使用提高图像的清晰度和质量。

**Base模型由U-Net，VAE，CLIP Text Encoder（两个）三个模块组成**，在FP16精度下Base模型大小6.94G（FP32：13.88G），其中U-Net大小5.14G，VAE模型大小167M以及两个CLIP Text Encoder一大一小分别是1.39G和246M。

**Refiner模型同样由U-Net，VAE，CLIP Text Encoder（一个）三个模块组成**，在FP16精度下Refiner模型大小6.08G，其中U-Net大小4.52G，VAE模型大小167M（与Base模型共用）以及CLIP Text Encoder模型大小1.39G（与Base模型共用）。

![SDXL model pipeline](/public/upload/SDXL/1.png)

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

### **2.3 VAE模型**

VAE模型（变分自编码器，Variational Auto-Encoder）是一个经典的生成式模型。在传统深度学习时代，GAN的风头完全盖过了VAE，但VAE简洁稳定的Encoder-Decoder架构，**以及能够高效提取数据Latent特征的关键能力**，让其跨过了周期，在AIGC时代重新繁荣。

Stable Diffusion XL依旧是**基于Latent**的扩散模型，**所以VAE的Encoder和Decoder结构依旧是Stable Diffusion XL提取图像Latent特征和图像像素级重建的关键一招**。

当输入是图片时，Stable Diffusion XL和Stable Diffusion一样，首先会使用VAE的**Encoder结构将输入图像转换为Latent特征**，然后U-Net不断对Latent特征进行优化，最后使用VAE的**Decoder结构将Latent特征重建出像素级图像**。除了提取Latent特征和图像的像素级重建外，**VAE还可以改进生成图像中的高频细节，小物体特征和整体图像色彩**。

当Stable Diffusion XL的输入是文字时，这时我们不需要VAE的Encoder结构，只需要Decoder进行图像重建。VAE的灵活运用，让Stable Diffusion系列增添了几分优雅。

**Stable Diffusion XL使用了和之前Stable Diffusion系列一样的VAE结构**，但在训练中选择了**更大的Batch-Size（256 vs 9）**，并且对模型进行指数滑动平均操作（**EMA**，exponential moving average），EMA对模型的参数做平均，从而提高性能并增加模型鲁棒性。

![img](/public/upload/SDXL/3.png)

在损失函数方面，使用了久经考验的**生成领域“交叉熵”—感知损失（perceptual loss）**以及回归损失来约束VAE的训练过程。

与此同时，VAE的**缩放系数**也产生了变化。VAE在将Latent特征送入U-Net之前，需要对Latent特征进行缩放让其标准差尽量为1，之前的Stable Diffusion系列采用的**缩放系数为0.18215，**由于Stable Diffusion XL的VAE进行了全面的重训练，所以**缩放系数重新设置为0.13025**。

注意：由于缩放系数的改变，Stable Diffusion XL VAE模型与之前的Stable Diffusion系列并不兼容。

### **2.4 CLIP Text Encoder模型**

**CLIP模型主要包含Text Encoder和Image Encoder两个模块**，在Stable Diffusion XL中，和之前的Stable Diffusion系列一样，**只使用Text Encoder模块从文本信息中提取Text Embeddings**。

**不过Stable Diffusion XL与之前的系列相比，使用了两个CLIP Text Encoder，分别是OpenCLIP ViT-bigG（1.39G）和OpenAI CLIP ViT-L（246M），从而大大增强了Stable Diffusion XL对文本的提取和理解能力。**

其中OpenCLIP ViT-bigG是一个只由Transformer模块组成的模型，一共有32个CLIPEncoder模块，是一个强力的特征提取模型。

OpenAI CLIP ViT-L同样是一个只由Transformer模块组成的模型，一共有12个CLIPEncoder模块。

OpenCLIP ViT-bigG的优势在于模型结构更深，特征维度更大，特征提取能力更强，但是其两者的基本CLIPEncoder模块是一样的。

**与传统深度学习中的模型融合类似**，Stable Diffusion XL分别提取两个Text Encoder的倒数第二层特征，并进行concat操作作为文本条件（Text Conditioning）。其中OpenCLIP ViT-bigG的特征维度为77x1280，而CLIP ViT-L的特征维度是77x768，所以输入总的特征维度是77x2048（77是最大的token数），再通过Cross Attention模块将文本信息传入Stable Diffusion XL的训练过程与推理过程中。

### **2.5 Refiner模型**

DeepFloyd和StabilityAI联合开发的DeepFloyd IF**是一种基于像素的文本到图像三重级联扩散模型**，大大提升了扩散模型的图像生成能力。

这次，Stable Diffusion XL终于也开始使用级联策略，在U-Net（Base）之后，级联Refiner模型，进一步提升生成图像的细节特征与整体质量。

**通过级联模型提升生成图片的质量，可以说这是AIGC时代里的模型融合。和传统深度学习时代的多模型融合策略一样，不管是学术界，工业界还是竞赛界，都是“行业核武”般的存在。**

![img](/public/upload/SDXL/4.png)

由于已经有U-Net（Base）模型生成了图像的Latent特征，所以**Refiner模型的主要工作是在Latent特征进行小噪声去除和细节质量提升**。

**Refiner模型和Base模型一样是基于Latent的扩散模型**，也采用了Encoder-Decoder结构，和U-Net兼容同一个VAE模型，不过Refiner模型的Text Encoder只使用了OpenCLIP ViT-bigG。

**下图是Stable Diffusion XL Refiner模型的完整结构图**

![img](/public/upload/SDXL/5.webp)

在Stable Diffusion XL推理阶段，输入一个prompt，通过VAE和U-Net（Base）模型生成Latent特征，接着给这个Latent特征加一定的噪音，在此基础上，再使用Refiner模型进行去噪，以提升图像的整体质量与局部细节。

![img](/public/upload/SDXL/6.png)

可以看到，**Refiner模型主要做了图像生成图像的工作**，其具备很强的**迁移兼容能力**，可以作为Stable Diffusion，GAN，VAE等生成式模型的级联组件，不管是对学术界，工业界还是竞赛界，无疑都是一个无疑都是一个巨大利好。

![img](/public/upload/SDXL/7.png)

只使用U-Net（Base）模型，Stable Diffusion XL模型的效果已经大幅超过SD1.5和SD2.1，当增加Refiner模型之后，Stable Diffusion XL达到了更加优秀的图像生成效果。

### 2.6 训练技巧&细节

Stable Diffusion XL在训练阶段提出了很多Tricks，包括图像尺寸条件化策略，图像裁剪参数条件化以及多尺度训练。**这些Tricks都有很好的通用性和迁移性，能普惠其他的生成式模型。**

推荐图片大小

- 21:9 – 1536 x 640
- 16:9 – 1344 x 768
- 3:2 – 1216 x 832
- 5:4 – 1152 x 896
- 1:1 – 1024 x 1024

### 实践

目前可以在本地运行SDXL的主流方法为

- **[AUTOMATIC1111's Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui)**最流行的 WebUI，拥有最多的功能和扩展，最容易学习。本说明书将使用此方法
- **[ComfyUI ](https://github.com/comfyanonymous/ComfyUI)**上手较难，node based interface，但生成速度非常快，比 AUTOMATIC1111 快 5-10 倍。允许使用两种不同的正向提示。

#### Install/Upgrade AUTOMATIC1111

下载base model 和 refiner model（每个～6GB）[huggingface网站](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

base model作为训练或推理时的底模使用

![img](/public/upload/SDXL/8.png)

refiner model仅在推理时使用

![Refiner model selector ](/public/upload/SDXL/9.png)

- **Checkpoint**: 选择refiner model

- **Switch at**: 切换位置，该值控制在哪一步切换到refiner model。例如，在 0.5 时切换并使用 40 个步骤表示在前 20 个步骤中使用base model，在后 20 个步骤中使用refiner model。

  如果切换到 1，则只使用base model。

经验最优值为**30 steps** and **switch at 0.6**
