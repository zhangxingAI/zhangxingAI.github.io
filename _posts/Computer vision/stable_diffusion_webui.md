layout: post
title: Stable Diffusion
category: 计算机视觉
tags: 
keywords:
typora-root-url: ../..

### Stable Diffusion

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

**Highres. fix** 使用两个步骤的过程进行生成，以较小的分辨率创建图像，然后在不改变构图的情况下改进其中的细节，选择该部分会有两个新的参数 **Scale latent** 在潜空间中对图像进行缩放。另一种方法是从潜在的表象中产生完整的图像，将其升级，然后将其移回潜在的空间。Denoising strength 决定算法对图像内容的保留程度。在0处，什么都不会改变，而在1处，你会得到一个不相关的图像；

**Batch count、 Batch size** 都是生成几张图，前者计算时间长，后者需要显存大；

**CFG Scale** 分类器自由引导尺度——图像与提示符的一致程度——越低的值产生越有创意的结果；

**Seed** 种子数，只要中子数一样，参数一致、模型一样图像就能重新；

**Denoising strength** 与原图一致性的程度，一般大于0.7出来的都是新效果，小于0.3基本就会原图缝缝补补；