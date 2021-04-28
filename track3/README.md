### this is the repo for track 3
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
python ./src/dialogue_system/run/run.py --train_mode 2 --saved_model=<model_dir> 
 ```

* 测试
```python
python ./src/dialogue_system/run/run.py --train_mode 0 --saved_model=<model_dir> 
 ```
 
注：该文件仅为提交程序的参考，提交时请提交整个模型
    有一些已经训练好的baseline存放在'./src/dialogue_system/model/dqn/checkpoint04/checkpoint_d4_agt1_dqn1_T22/'下，可用于参考

- DQN 模型
* 训练
```python
python ./src/dialogue_system/run/run.py --train_mode 1
 ```
参数说明见train.py的args和dialogue_configuration.py，可以更改各种情况的reward

* 验证
```python
python ./src/dialogue_system/run/run.py --train_mode 2 --saved_model=<model_dir> 
 ```

* 测试
```python
python ./src/dialogue_system/run/run.py --train_mode 0 --saved_model=<model_dir> 
 ```
 
注：该文件仅为提交程序的参考，提交时请提交整个模型

- SVM&MLP 模型
```python
python ./src/classifier/run/run.py --model SVM --fold False 
 ```
注：如需使用MLP模型，则将SVM改为MLP；fold 参数控制将数据集所有按5-fold划分(True)或按指定划分(False)
因数据集中没有test set，故可能与报告数据有所出入

# 模型
关于baseline的一些模型结果存放在[Chatbox_baseline实验结果.xlsx](./Chatbox_baseline实验结果.xlsx) 中

# References

- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
- [abisee/pointer_summarizer]([GitHub - abisee/pointer-generator: Code for the ACL 2017 paper "Get To The Point: Summarization with Pointer-Generator Networks"](https://github.com/abisee/pointer-generator))

