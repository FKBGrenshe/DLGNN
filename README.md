# DLGNN



训练 -- train.py

1. 权重共享、滑动窗口机制未实现
2. algorithm 文件 对应 论文中的 算法流程
3. readoutlayer 在论文中模型结构图中有提及，但没找到详细说明
4. 数据处理流程：
   1. 从数据集中提取数据 -- 在dataProcess/dataPreprocess.py的step函数中
   2. 填充全零数据 -- placeholder -- 在dataProcess/dataPreprocess.py的step函数中 参数代表填充到多长
   3. 基于数据 生成图 - 基于图 生成线图 -- 在dataProcess/GraphBuild.py中
5. 训练文件 train.py 