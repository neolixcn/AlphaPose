# Alphapose 模型部署指南

## 1. 模型转 trt engine
alphapose模型本身的介绍可以参考这篇[论文](https://arxiv.org/abs/1612.00137)，以及对应的 [git repo]()。如果只考虑需要部署，可以直接使用我们的 [google lab](https://colab.research.google.com/drive/10Oq1S9PC6zU5Qzeri0ZiFtJjeIb-ynjF?usp=sharing)
环境，执行完毕即可得到 alphapose_neolix.onnx 文件。现有的已经转换好的模型存放在服务器：/nfs/nas/model_release/alphapose/alphapose.onnx 上。
由于 alphapose 模型本身的算子在trt中都支持，所以从 onnx 转换 engine 还比较方便，fp32 和 fp16 模型，可以通过两种方式获得：

#### (1) trtexec 工具转换，使用如下指令：

``` bash
#生成fp32 engine
./trtexec --onnx=alphapose.onnx --saveEngine=alphapose_fp32.engine 

#生成fp16 engine
./trtexec --onnx=alphapose.onnx --saveEngine=alphapose_fp16.engine --fp16
```

#### (2) 代码的方式进行转换 
参考 [perception-model @ neolix](https://github.com/neolixcn/perception-models/tree/master/tools/TensorRT_implementer) 的readme 文档，
修改 config.h 中的配置参数，如输入输出名字，模型精度，onnx模型位置等即可。


## 2. 模型量化
在 fp16 engine 的推理速度仍然不能达到我们的要求的情况下，量化是必要的。
量化的第一步需要选择校准集，对于 alphapose 这类以图像为输入的网络，一般选取100张真实情况下会得到的输入图像，
具有多样性（如，各种手势，各种角度，各种距离，etc）。在真实场景中模型的输入为yolo检测后的输入，然后resize到统一尺寸 (3,256,192)。
因此我们在校准集中也要选择相同的数据，并且在之后读入量化数据时对图片进行resize。下图展示了几张校准集中的图片。


<div align="center">
    <img src="neolix_readme_imgs/quanti_data_examples_no_face.png", width="700">
</div>

在量化过程中，需要
#### (1) 设置 config.h 中的量化相关路径与参数，主要是如下几项：
``` cpp
static std::string ModelPrecision = "int8";
static std::string ModelPath = "/path/to/your/model/alphapose.onnx";      
/***      
 * model_name 模型网络类型      
 * calib_table_file: 量化表路径 
 * calibration_batch_size: 生成量化表的模型前向batch_size
 * calib_data_path: 模型量化用到的数据所在文件夹
 * calib_data_list: 模型量化用到的数据名
 */
static std::string model_name = "alphapose";
static std::string calib_table_file = "CalibrationTable";
static int calibration_batch_size = 1;
static std::string calib_data_path = "/path/to/your/calibset/";
static std::string calib_data_list = "calib_set.txt";
```
其中需要注意的是，在calib_data_list项中，需要准备一个包含了校准集中所有名字的 txt 文件（在上例中为 “calib_set.txt”），每个文件名按空行分开。

#### (2) 修改 calibrator.cpp 中加载校准数据的部分代码
主要是 
``` cpp
boolInt8EntropyCalibrator::getBatch(void **bindings, constchar **names, int nbBindings)
```
函数中，关于图片的预处理的操作，需要修改如图片尺寸为 (3,256,192)，normalize的参数为 mean 为 {0,0,0}， std 为 {1,1,1}。

然后编译运行 ./Test_Tensorrt，等待程序结束即可在 onnx 模型的同一目录下得到 alphapose_int8.engine，此即可以部署到线上 pipeline的 engine 模型。同时，量化表 CalibrationTable 文件也会被生成，有了这个文件，下次量化时就不需要校准集数据了。
