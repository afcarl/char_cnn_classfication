import torch

# Parameters
# ==================================================
ltype = torch.cuda.LongTensor
ftype = torch.cuda.FloatTensor

# Model Switch
Large = 1 #  0:Small / 1:Large
Mode = 0 # 0:LowerOnly / 1:Full / 2:Lookup / 3:W2V

# Model Hyperparameters
# Cnn Model
cnn_w = [7,7,3,3,3,3]
pool_w = [3,3,3]
large_feature = 1024
small_feature = 256
l_0 = 256 if Mode==3 else 1014
cnn_output_feature = large_feature if Large else small_feature ### Large/Small
# FC Model
output_large = 2048
output_small = 1024
l_6 = int((l_0-96)/27)
fc_input_feature = l_6*cnn_output_feature
fc_hidden_feature = output_large if Large else output_small ### Large/Small
# Weight init
weight_m = 0
weight_v = 0.02 if Large else 0.05

# Target Class
class_n = 2

uniq_char = 71+26 if Mode==1 else 71 ### Full/W2V
max_char = 256 if Mode==3 else 1014 ### W2V
char_dim = 300 if Mode==3 else 96 if Mode==1 else 70 # Full/W2V

# Training Parameters
batch_size = 128
num_epochs = 30
learning_rate = 0.01
momentum = 0.9
evaluate_every = 3
