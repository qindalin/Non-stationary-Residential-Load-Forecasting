import copy
import torch
import numpy as np
import pandas as pd
import os
from utils.trainer import Trainer
from models.DLinear import DLinear
from models.MLP import MLP
from models.LSTM import LSTM
from models.CNN import CNN
from models.PatchTST import PatchTST
from utils.utils import init_dl_program
from tqdm import tqdm
from utils.utils import pkl_save

device = "cuda" if torch.cuda.is_available() else "cpu"
model_list = {'DLinear': DLinear,
              'MLP': MLP,
              'LSTM': LSTM,
              'CNN': CNN,
              'PatchTST': PatchTST}

class Server():

    def __init__(self,
                 clients,
                 args) -> None:
        self.args = args
        self.clients = clients
        self.model_name = self.clients[0].model_name
        self.device = init_dl_program(self.args.gpu, seed=self.args.seed, max_threads=self.args.max_threads)
        self.args.device = self.device
        if self.model_name == 'PatchTST':
            self.args.enc_in = getattr(self.clients[0].args, 'enc_in', self.args.enc_in)
        self.global_model = model_list[self.model_name](configs=self.args).to(self.device)
        self.path = f"save/offline federated/{self.model_name}"
        os.makedirs(self.path, exist_ok=True)

    def set_model(self, model_params):
        strict = self.model_name != 'PatchTST'
        self.global_model.load_state_dict(model_params, strict=strict)

    def offline_fed(self):
        # Initialize
        initial_params = self.global_model.state_dict()
        for client in self.clients:
            client.set_model(initial_params)

        # Local train
        for e in range(self.args.fed_rounds):
            print(f'Federated training [Epoch {e}/{self.args.fed_rounds}]')
            if self.args.fed_type == 'fedsgd':
                client_updates = []
                initial_global_params = self.global_model.state_dict()
                for client in self.clients:
                    client.train(epochs=self.args.fed_local_epochs, save=False)
                    delta = {name: param.data.cpu() - initial_global_params[name].cpu() for name, param in client.global_model.named_parameters()}
                    client_updates.append(delta)
                avg_delta = self.model_aggregate(client_updates)
                global_params = {name: initial_global_params[name] + self.args.fedsgd_lr * avg_delta[name].to(self.device) for name in avg_delta}

            elif self.args.fed_type == 'fedavgm':
                client_updates = []
                initial_global_params = self.global_model.state_dict()
                if not hasattr(self, 'momentum'):
                    self.momentum = {k: torch.zeros_like(v) for k, v in initial_global_params.items()}
                for client in self.clients:
                    client.train(epochs=self.args.fed_local_epochs, save=False)
                    delta = {name: param.data.cpu() - initial_global_params[name].cpu() for name, param in client.global_model.named_parameters()}
                    client_updates.append(delta)
                avg_delta = self.model_aggregate(client_updates)
                for name in avg_delta:
                    self.momentum[name] = self.args.fedavgm_beta * self.momentum[name].to(self.device) + avg_delta[name].to(self.device)
                    global_params = {
                        name: initial_global_params[name] + self.args.fedavgm_lr * self.momentum[name].to(self.device)
                        for name in avg_delta
                    }
            else:    
                client_updates = []
                for client in self.clients:
                    client.train(epochs=self.args.fed_local_epochs, save=False)
                    client_updates.append(client.global_model.state_dict())
                # Model average   
                global_params = self.model_aggregate(client_updates)
                
            self.global_model.load_state_dict(global_params)

            # Model distribute
            for client in self.clients:
                client.set_model(global_params)
        if self.args.fed_type is None:
            pkl_save(f'{self.path}/model.pkl', self.global_model)
        else:
            pkl_save(f'{self.path}/{self.args.fed_type}_model.pkl', self.global_model)
        return self.global_model

    def online_fed(self, decay_update=False, adaptive=True):
        global_params = self.global_model.state_dict()
        for client in self.clients:
            client.set_model(global_params)
            client.init_weight()
        
        test_timesteps = len(self.clients[0].test_loader.dataset)
        for timestep in tqdm(range(test_timesteps), desc="Timesteps"):
            if timestep % self.args.pred_len == 0:
                # print(f'Online Federated training [Timestep {timestep}/{test_timesteps}]')
                if self.args.fed_type == 'fedsgd':
                    client_updates = []
                    initial_global_params = self.global_model.state_dict()
                    for client in self.clients:
                        client.global_local_online_update(timestep, decay_update=decay_update, adaptive=adaptive)
                        delta = {name: param.data.cpu() - initial_global_params[name].cpu() for name, param in client.global_model.named_parameters()}
                        client_updates.append(delta)
                    avg_delta = self.model_aggregate(client_updates)
                    global_params = {name: initial_global_params[name] + self.args.fedsgd_lr * avg_delta[name].to(self.device) for name in avg_delta}
                
                elif self.args.fed_type == 'fedavgm':
                    client_updates = []
                    initial_global_params = self.global_model.state_dict()
                    if not hasattr(self, 'momentum'):
                        self.momentum = {k: torch.zeros_like(v) for k, v in initial_global_params.items()}
                    for client in self.clients:
                        client.global_local_online_update(timestep, decay_update=decay_update, adaptive=adaptive)
                        delta = {name: param.data.cpu() - initial_global_params[name].cpu() for name, param in client.global_model.named_parameters()}
                        client_updates.append(delta)
                    avg_delta = self.model_aggregate(client_updates)
                    for name in avg_delta:
                        self.momentum[name] = self.args.fedavgm_beta * self.momentum[name].to(self.device) + avg_delta[name].to(self.device)
                        global_params = {
                            name: initial_global_params[name] + self.args.fedavgm_lr * self.momentum[name].to(self.device)
                            for name in avg_delta
                        }
                else:
                    client_updates = []
                    for client in self.clients:
                        client.global_local_online_update(timestep, decay_update=decay_update, adaptive=adaptive)
                        client_updates.append(client.global_model.state_dict())
                    global_params = self.model_aggregate(client_updates)

                self.global_model.load_state_dict(global_params)
                for client in self.clients:
                    client.set_model(global_params)
    
    def model_aggregate(self, client_updates):
        avg_params = {}
        for key in client_updates[0].keys():
            stacked = torch.stack([update[key] for update in client_updates], dim=0)
            avg_params[key] = torch.mean(stacked, dim=0)
        return avg_params
    
    def local_train(self):
        print('Launch Local Training!')
        for client in self.clients:
            client.train()
