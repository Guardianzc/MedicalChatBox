# Requirement

python=3.6 

torch==1.2.0

tensorflow==1.13.1

pandas==0.25.1

sumeval==0.2.2

其他依赖包和版本可以参考 requirement.txt

# 数据

* 数据说明

训练集和验证集参赛选手可以重新进行划分。但是测试集是固定的。

参赛选手可以选择使用赛道一的数据改进模型。但是要求测试时的输入为主诉和对话。（比赛第二阶段会检查模型）

* 数据预处理

```
python make_datafiles.py 
```

处理好后的文件会存在medi_finished_dir 文件夹中，其中vocab是基于训练集中获取的字典。file_names_train，file_names_dev和file_names_test保存了example id。

# seq2seq模型

- 训练

 ```python
python train.py --use_gpu --exp_name=s2s 
 ```

参数说明见train.py的args，可以使用python train -h 进行查看。

* 验证

```python
python decode.py --model_filename=<model_dir> --decode_filename=medi_finished_dir/dev.bin --mode=dev --compute_rouge  --output_filenames=medi_finished_dir/file_names_dev
```

<model_dir> 中填写模型的地址

* 测试

```python
python decode.py --model_filename=<model_dir>
```

生成用于测试的文件。将“生成摘要”文件夹压缩打包，作为提交评测的文件。

# pointer_generator模型

* 训练

```python
python train.py --use_gpu --pointer_gen --is_coverage --exp_name=pg
```

* 验证

``` python
python decode.py --model_filename=<model_dir> --decode_filename=medi_finished_dir/dev.bin --mode=dev --compute_rouge  --output_filenames=medi_finished_dir/file_names_dev --pointer_gen --is_coverage 
```

* 测试

```python
python decode.py --model_filename=<model_dir> --pointer_gen --is_coverage
```



# 第一阶段评测说明

在文件夹evaluation中，给出了第一阶段提交的文件样例和自动化评价的代码。

- 提交文件要求：对命名为“生成摘要_test" 的文件夹打包成 "生成摘要\_test.zip" 上传。其中含有以“test_XXXX"命名的文件，XXXX是example_id。每个文件是生成的诊疗报告。其中每个字符用空格‘ ’隔开。

- 自动化测评的代码：evaluate.py



# References

- [Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)
- [abisee/pointer_summarizer]([GitHub - abisee/pointer-generator: Code for the ACL 2017 paper "Get To The Point: Summarization with Pointer-Generator Networks"](https://github.com/abisee/pointer-generator))

