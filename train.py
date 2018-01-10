#! /usr/bin/env python

import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import config
import models
from data_loader import Data
import sys

# Type Parameters
ltype = config.ltype
ftype = config.ftype
# Training Parameters
learning_rate = config.learning_rate

def parameters(*argv):
    params = []
    for model in argv:
        params += list(model.parameters())

    return params

def print_score(batches, test_len, step):
    batch_hc = 0. # hit count
    for batch in batches:
        star_batch, text_batch, len_batch = zip(*batch)
        batch_hc += run(star_batch, text_batch, len_batch, step=step)
    print("Validation Error :", round((1 - batch_hc/test_len)*100, 2), datetime.datetime.now())

##############################################################################################
def run(stars, texts, lens, step):

    optimizer.zero_grad()

    stars = Variable(torch.from_numpy(np.asarray(stars)).type(ltype))
    texts = torch.from_numpy(np.asarray(texts)).type(ltype) # batch x max_char

    # Lookup
    if config.Mode == 2 or config.Mode == 3:
        texts = Variable(texts)
    else:
        # One-hot encoding
        text_onehot = []
        for text in texts:
            onehot = torch.zeros(config.uniq_char, config.max_char).type(ftype)
            onehot.scatter_(0, text.view(1,-1), 1)
            text_onehot.append(onehot[1:].view(1,config.uniq_char-1,config.max_char))
        # batch x uniq_char x max_char
        texts = Variable(torch.cat(text_onehot, 0))

    # CNN
    cnn_output = cnn_models(texts)
    # FC
    fc_output = linear_model(cnn_output)

    if (step > 1):
        _, max_idx = torch.max(fc_output, dim=1)
        return torch.sum(max_idx == stars).data.cpu().numpy()[0]

    # CrossEntropyLoss
    loss = loss_model(fc_output, stars)

    loss.backward()
    optimizer.step()
    
    return loss.data.cpu().numpy()[0]

##############################################################################################
##############################################################################################
if __name__ == "__main__":

    # Data Preparation
    data = Data(config.Mode)
    data.load()
    config.uniq_char = len(data.char_list)

    # Model Preparation
    cnn_models = models.Conv1d(config.cnn_w, config.pool_w, config.cnn_output_feature, data.char_dict).cuda()
    linear_model = models.FC(config.fc_input_feature, config.fc_hidden_feature, config.class_n).cuda()
    loss_model = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(parameters(cnn_models, linear_model), lr=learning_rate, momentum=config.momentum)

    for i in range(config.num_epochs):
        # Training
        batch_loss = 0.
        train_batches = data.train_batch_iter(config.batch_size)
        for j, train_batch in enumerate(train_batches):
            star_batch, text_batch, len_batch = zip(*train_batch)
            batch_loss += run(star_batch, text_batch, len_batch, step=1)
            if (j+1) % 1000 == 0:
                print("batch #{:d}: ".format(j+1), "batch_loss :", batch_loss/j, datetime.datetime.now())

        # Validation 
        if (i+1) % config.evaluate_every == 0:
            linear_model.eval()
            print("==================================================================================")
            print("Evaluation at epoch #{:d}: ".format(i+1))
            validation_batches = data.test_batch_iter(config.batch_size)
            print_score(validation_batches, len(data.star_test), step=2)
            learning_rate /= 2
            optimizer = torch.optim.SGD(parameters(cnn_models, linear_model), lr=learning_rate, momentum=config.momentum)
            linear_model.train()
