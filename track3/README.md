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
参数说明见train.py的args和dialogue_configuration.py，可以更改各种情况的reward

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
关于提交，请参赛者提交两部分文件：  
    1、我们要求所有参赛者提交模型和代码，在[评测文件夹](./MedicalChatbot-track3//Evaluation/)中，我们提供了用户模拟器、测试集示例、输出结果示例和评测示例，参赛者可以import我们所提供的用户模拟器进行交互，具体方法请参见其中代码
       在提交的模型和代码中，请提交一个包含以下字段的可运行脚本：
	Python run.py –data_path (数据集路径) —model_path (模型路径)


# 评测
评测相关程序位于[评测文件夹](./MedicalChatbot-track3//Evaluation/) 中，通过模型输出的 result.json 和 goal_set 进行正确率和F1的计算
 
result.json的示例和评测文件也保存在	[评测文件夹](./MedicalChatbot-track3//Evaluation/) 中

# References

- [Task-oriented Dialogue System for Automatic Diagnosis](http://www.aclweb.org/anthology/P18-2033)
