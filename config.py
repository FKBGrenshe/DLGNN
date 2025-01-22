# 配置文件，定义超参数和其他参数
CONFIG = {
    'learning_rate': 0.01,
    'k_GCNII':10,  # 10层 GCNII
    'k_GCNII_GRU':2,  # 2层 GCNII-GRU
    'hid_dim':50,
    'weight_decay':0.001,
    'sequence_length':100,
    'slidingWindowOverlap':0.5,
    'weightSharingWindowSize':5
    # 'num_epochs': 100,
    # 'batch_size': 32,
    # 'model_save_path': 'model.pth',
}
