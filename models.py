import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class Conv1d(nn.Module):
    def __init__(self, cnn_kernel_sizes, pool_kernel_sizes, out_channels, weight_init):
        super(Conv1d, self).__init__()

        # embedding:
        self.char_weight = nn.Embedding(config.uniq_char, config.char_dim, padding_idx=0)
        self.char_weight.weight.data.normal_(config.weight_m, config.weight_v)
        if config.Mode == 3:
            self.char_weight.weight.data.copy_(torch.from_numpy(weight_init))
        # attributes:
        self.mode = config.Mode
        self.l_0 = config.max_char
        self.in_channels = config.char_dim
        self.out_channels = out_channels
        self.cnn_kernel_sizes = cnn_kernel_sizes
        self.pool_kernel_sizes = pool_kernel_sizes

        # modules:
        self.cnn1 = nn.Conv1d(self.in_channels, self.out_channels, self.cnn_kernel_sizes[0])
        self.cnn2 = nn.Conv1d(self.out_channels, self.out_channels, self.cnn_kernel_sizes[1])
        self.cnn3 = nn.Conv1d(self.out_channels, self.out_channels, self.cnn_kernel_sizes[2])
        self.cnn4 = nn.Conv1d(self.out_channels, self.out_channels, self.cnn_kernel_sizes[3])
        self.cnn5 = nn.Conv1d(self.out_channels, self.out_channels, self.cnn_kernel_sizes[4])
        self.cnn6 = nn.Conv1d(self.out_channels, self.out_channels, self.cnn_kernel_sizes[5])

        self.maxpool1 = nn.MaxPool1d(self.pool_kernel_sizes[0])
        self.maxpool2 = nn.MaxPool1d(self.pool_kernel_sizes[1])
        self.maxpool3 = nn.MaxPool1d(self.pool_kernel_sizes[2])

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                m.weight.data.normal_(config.weight_m, config.weight_v)

    def forward(self, chars):
        # Lookup
        if self.mode == 2 or self.mode == 3:
            chars = self.char_weight(chars).permute(0,2,1)

        conv_out = self.cnn1(chars)
        pool_out = F.relu(self.maxpool1(conv_out))
        conv_out = self.cnn2(pool_out)
        pool_out = F.relu(self.maxpool2(conv_out))
        conv_out = F.relu(self.cnn3(pool_out))
        conv_out = F.relu(self.cnn4(conv_out))
        conv_out = F.relu(self.cnn5(conv_out))
        conv_out = self.cnn6(conv_out)
        pool_out = F.relu(self.maxpool3(conv_out))
        return pool_out

class FC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FC, self).__init__()

        # attributes:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # modules:
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(config.weight_m, config.weight_v)

    def forward(self, cnn_output):
        cnn_output = cnn_output.view(-1, self.input_size)
        fc_output = F.relu(F.dropout(self.linear1(cnn_output)))
        fc_output = F.relu(F.dropout(self.linear2(fc_output)))
        fc_output = self.linear3(fc_output)
        return fc_output
