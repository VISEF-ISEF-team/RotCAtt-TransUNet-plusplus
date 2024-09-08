import ml_collections

def get_config():
    config = ml_collections.ConfigDict()
    
    # Input
    config.num_classes = 12
    config.in_channels = 1
    config.img_size    = 256
    config.p           = [16, 8, 4]
    
    # Dense
    config.dense_filters = [1, 64, 128, 256]
    
    # Rotatory attention
    config.df          = [64, 128, 256]
    config.dk          = [64, 128, 256]
    config.dv          = [64, 128, 256]
    config.dq          = [64, 128, 256]
    
    # Transformer
    config.num_heads = 4
    config.num_layers = 4
    config.mlp_ratio = 4
    config.dropout_rate = 0.1
    
    # DecoderCup
    config.head_channels = 512
    
    # Other
    config.vis = True
    
    return config