from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import openood.utils.comm as comm


class BasePostprocessor:
    def __init__(self, config):
        self.config = config

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        device = next(net.parameters()).device  # Get the device of the model's parameters
        data = data.to(device)
        outputs = net(data)
        score = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self,
                  net: nn.Module,
                  data_loaders_dict: dict,
                  progress: bool = True):
                
        device = next(net.parameters()).device 
        pred_list, conf_list, label_list = [], [], []
        for loader_name, data_loader in data_loaders_dict.items():
          for batch in tqdm(data_loader,
                            disable=not progress or not comm.is_main_process()):
              data = batch[0].to(device)
              label = batch[1].to(device)
              pred, conf = self.postprocess(net, data)

              pred_list.append(pred.cpu())
              conf_list.append(conf.cpu())
              label_list.append(label.cpu())

        # convert values into numpy array
        pred_list = torch.cat(pred_list).numpy().astype(int)
        conf_list = torch.cat(conf_list).numpy()
        label_list = torch.cat(label_list).numpy().astype(int)

        return pred_list, conf_list, label_list
