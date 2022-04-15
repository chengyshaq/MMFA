# -*- coding: UTF-8 -*-
import torch.optim as optim
from torch.utils.data import DataLoader
from utilities.common_tools import ViewsDataset

from model.view_block import ViewBlock
from model.mmfa_model import MMFA_Model
from model.train_mmfa_model import Train_MMFA_Model


def main_mmfa_model_train(features, labels, args,
                     loss_coefficient,
                           model_args = None, weight_decay=1e-5, fold=1):
    # step 1: load views features and labels
    views_dataset = ViewsDataset(features, labels)
    views_data_loader = DataLoader(views_dataset, batch_size=64, shuffle=True,
                                   num_workers=0)
    # step 2: instantiation View Model
    label_nums = labels.shape[1]
    view_code_list = list(features.keys())
    view_feature_nums_list = [features[code].shape[1] for code in view_code_list]
    view_blocks = [ViewBlock(view_code_list[i], view_feature_nums_list[i],
                             args['comm_feature_nums']) for i in range(len(view_code_list))]
    mmfa_model = MMFA_Model(view_blocks, args['comm_feature_nums'],view_feature_nums_list,
                             label_nums, model_args)

    # step 3: init optimizer
    optimizer = optim.Adam(mmfa_model.parameters(), lr=0.01, weight_decay=weight_decay)
   
    # step 4: train model
    trainer = Train_MMFA_Model(mmfa_model, views_data_loader, args['epoch'],
                                optimizer, args['show_epoch'], loss_coefficient,
                                args['model_save_epoch'], args['model_save_dir'])
    loss_list = trainer.train(fold)
    return loss_list