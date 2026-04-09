用户文档：proj_RATR_complex

models：保存模型权重文件

	ratr_complex_model.pth    雷达目标识别复杂模型权重文件

dataset：测试数据，各提供了100帧

ratr_raw：输入和输出结果文件夹

    input 里面有代码需要的json参数文件
    result 是输出的识别结果混淆矩阵

proj_RATR_complex:主程序，配置好虚拟环境后即可运行

environment.yml 是conda文件移植环境，python版本3.10.18，CUDA 12.8。
如果不同版本CUDA需要去官网专门安装对应的torch的版本，此外如果环境移植不正确，也可根据提供的yml的依赖库从官网配置环境。