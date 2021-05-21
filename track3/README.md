# 赛道3材料介绍
# Requirement

python==3.6.4
tensorflow==1.3
numpy==1.13.3

其他依赖包和版本可以参考 requirement.txt

# 数据

* 数据说明

训练集和验证集参赛选手可以重新进行划分。但是测试集是固定的。
数据集请在下载后存放在'./src/dialogue_system/data/dataset/label/' 下

# 模型

- DQN 模型
* 训练
```python
python ./src/dialogue_system/run/run.py --train_mode 1
 ```
参数说明见run.py的args和dialogue_configuration.py，可以更改各种情况的reward

* 验证
```python
python ./src/dialogue_system/run/run.py --train_mode 2 --saved_model=<model_dir> --print_result 1
 ```
print_result 参数决定是否输出.json的结果文件
* 测试
```python
python ./src/dialogue_system/run/run.py --train_mode 0 --saved_model=<model_dir> --print_result 1
 ```
 
注：该文件仅为提交程序的参考，提交时请提交整个模型
    有一些已经训练好的baseline存放在'./src/dialogue_system/model/dqn/checkpoint04/checkpoint_d4_agt1_dqn1_T22/'下，可用于参考


# 提交

1. repo文件夹

该文件夹至少需包括4个文件（文件夹）

- `models`：储存训练好的模型参数

- `user_simulator.py`：官方提供的用户模拟器，请勿修改！

  - 初始化参数为测试集输入数据路径（利用pickle储存的字典，参考`Evaluation/goal_set_simul.p`）

- `predict.py`：标签预测脚本，根据输入文件和模型，预测id对应的标签。需包含以下几个输入参数（**请统一用该参数名称！**）

  - `test_input_file`：测试集输入文件路径（pickle文件，参考`Evaluation/goal_set_simul.p`）
  - `test_output_file`：测试集输出文件路径（json文件，参考参考`Evaluation/result.json`）
  - `model_dir`：加载的模型存放的路径

- `run.sh`：执行预测脚本

  ```shell
  #!/bin/sh

  source ~/env/scv0xxx-3/bin/activate # 指定启用环境, 修改xxx即可

  python predict.py --test_input_file {test_file_path} --test_output_file {output_file_path} --model_dir ./models/best.pt
  ```

  **注：**`run.sh`中请修改`--model_dir`参数，指定模型路径，`--test_input_file`和`--test_output_file`保留`{test_file_path}`和`{output_file_path}`不要更改！

注：

提交前请对`predict.py`脚本进行测试！


# 评测
评测相关程序位于[评测文件夹](./MedicalChatbot-track3//Evaluation/) 中，通过模型输出的 result.json 和 goal_set 进行疾病预测正确率和症状预测Recall的计算  
每个疾病对用户模拟器进行症状问询的轮次不应大于11轮  
 总分 = 0.8 * 疾病判断的正确率 + 0.2 * 症状判断的召回率  
result.json的示例和评测文件也保存在	[评测文件夹](./MedicalChatbot-track3//Evaluation/) 中

# References

- [Task-oriented Dialogue System for Automatic Diagnosis](http://www.aclweb.org/anthology/P18-2033)
