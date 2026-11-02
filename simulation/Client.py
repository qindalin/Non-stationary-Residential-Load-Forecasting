import os
import copy
import torch
import numpy as np
import torch.nn as nn
import pandas as pd

from models.DLinear import DLinear
from models.Ensemble_MLP import Ensemble_MLP
from models.MLP import MLP
from models.LSTM import LSTM
from models.CNN import CNN
from models.PatchTST import PatchTST
from utils.utils import init_dl_program
from data_provider.data_factory import data_provider
from utils.trainer import Trainer, Ensemble_Trainer
from utils.utils import get_tensor_batch
from tqdm import tqdm
from torch import optim

device = "cuda" if torch.cuda.is_available() else "cpu"
model_list = {'DLinear': DLinear,
              'MLP': MLP,
              'LSTM': LSTM,
              'CNN': CNN,
              'PatchTST': PatchTST}

class Client():

    @staticmethod
    def _infer_enc_in(root_path, data_name, default_value):
        data_path = os.path.join(root_path, f'{data_name}.csv')
        if not os.path.exists(data_path):
            return default_value
        cols = pd.read_csv(data_path, nrows=0).columns.tolist()
        feature_cols = [col for col in cols if col != 'date']
        return len(feature_cols) if feature_cols else default_value

    def __init__(
            self,
            data_name,
            args,
            model_name = 'DLinear') -> None:

        self.data_name = data_name
        self.model_name = model_name
        self.args = args
        
        self.device = init_dl_program(self.args.gpu, seed=self.args.seed, max_threads=self.args.max_threads)
        self.args.device = self.device
        self.args.data_path = f'{data_name}.csv'
        if self.model_name == 'PatchTST':
            self.args.enc_in = self._infer_enc_in(self.args.root_path, data_name, self.args.enc_in)
        self.local_model = model_list[self.model_name](configs=self.args).to(self.device)
        self.global_model = model_list[self.model_name](configs=self.args).to(self.device)
        self.short_ensembler = Ensemble_MLP(n_inputs=args.pred_len * 3,
                             n_outputs=1, mlp_width=32,
                             mlp_depth=3, 
                             mlp_dropout=0.1, 
                             act=nn.Tanh()).to(self.device)
        self.long_weight = torch.zeros(1, device = self.device)
        self.long_weight.requires_grad = True
        self.short_weight = torch.zeros(1, device = self.device)
        self.global_decay_rate = nn.Parameter(torch.tensor(self.args.decay_rate, device=self.device))
        self.local_decay_rate = nn.Parameter(torch.tensor(self.args.decay_rate, device=self.device))
        # self.global_decay_rate = nn.Parameter(torch.zeros(1, device=self.device))
        # self.local_decay_rate = nn.Parameter(torch.zeros(1, device=self.device))
        self.path = f"save/{self.data_name}/{self.model_name}"
        self.train_data, self.train_loader = self._get_data(flag='train')
        _, self.vali_loader = self._get_data(flag='val')
        _, self.test_loader = self._get_data(flag='test')
        self.scaler = self.train_data.scaler

        self.local_trainer = Trainer(model=self.local_model, 
                                     args=self.args, 
                                     scaler=self.scaler,
                                     device=self.device, 
                                     path=self.path,
                                     model_name=self.model_name)
        self.global_trainer = Trainer(model=self.global_model, 
                                      args=self.args, 
                                      scaler=self.scaler,
                                      device=self.device, 
                                      path=self.path,
                                      model_name=self.model_name)
        self.ensemble_trainer = Ensemble_Trainer(ensembler=self.short_ensembler,
                                                 long_weight=self.long_weight,
                                                 short_weight=self.short_weight,
                                                 args=self.args,
                                                 device=self.device,
                                                 scaler=self.scaler,
                                                 model_name=self.model_name)
        self.online_local_preds = []
        self.online_global_preds = []
        self.online_inc_local_preds = []
        self.online_inc_global_preds = []
        self.online_ensemble_preds = []
        self.online_global_local_error_ratio = 0.5
        self.online_trues = []
        self.local_model_buffer = []
        self.global_model_buffer = []
        self.online_long_weight = []
        self.online_short_weight = []
        self.online_weight = []
        self.online_local_decay_rates = []
        self.online_global_decay_rates = []

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader
    
    def init_weight(self):
        self.short_weight = torch.zeros(1, device = self.device)
        self.long_weight = torch.zeros(1, device = self.device)
        self.long_weight.requires_grad = True
        self.ensemble_trainer.short_weight = self.short_weight
        self.ensemble_trainer.long_weight = self.long_weight

    def get_model(self, type):
        if type == 'local':
            model = self.local_model
        elif type == 'global':
            model = self.global_model
        return model
    
    def set_model(self, model_params, type='global'):
        strict = self.model_name != 'PatchTST'
        if type == 'local':
            self.local_model.load_state_dict(model_params, strict=strict)
            self.local_trainer.model = self.local_model
        if type == 'global':
            self.global_model.load_state_dict(model_params, strict=strict)
            self.global_trainer.model = self.global_model
    
    def train(self, epochs=None, save=False, type='global'):
        if type == 'local':
            trainer = self.local_trainer
        elif type == 'global':
            trainer = self.global_trainer
        trainer.train(train_loader=self.train_loader, vali_loader=self.vali_loader, epochs=epochs, save=save)
        # test_loss = self.trainer.vali(self.test_loader)
        # print(test_loss)

    def train_ensemble(self, epochs=None):
        self.ensemble_trainer.train(train_loader=self.train_loader,
                                    local_model=self.local_model,
                                    global_model=self.global_model,
                                    epochs=epochs)    

    def test(self, rescale=False, online=False, decay_update=False, type='local'):
        if type == 'local':
            model = self.local_model
            trainer = self.local_trainer
            model_buffer = self.local_model_buffer
            online_preds = self.online_local_preds
            online_inc_preds = self.online_inc_local_preds
        elif type == 'global':
            model = self.global_model
            trainer = self.global_trainer
            model_buffer = self.global_model_buffer
            online_preds = self.online_global_preds
            online_inc_preds = self.online_inc_global_preds

        for i, batch in enumerate(tqdm(self.test_loader)):
            if i % self.args.pred_len == 0:
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                if online:
                    inc_ensemble_output = self.incremental_model_ensemble(batch, decay_update=decay_update, type=type)
                    online_inc_preds.append(inc_ensemble_output.detach().cpu())
                    outputs = trainer.online_step(batch)
                    model_buffer.append(copy.deepcopy(trainer.model.state_dict()))
                    if len(model_buffer) > self.args.model_buffer_size:
                        model_buffer.pop(0)
                else:
                    model.eval()
                    with torch.no_grad():
                        outputs = trainer.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark, model)

                pred = outputs.detach().cpu().numpy() 
                true = batch_y[:, -self.args.pred_len:, -1].detach().cpu().numpy() 
                if rescale:
                    pred = self.scaler.inverse_transform(pred.reshape(-1,1)).reshape(batch_y.shape[0],-1,1)
                    true = self.scaler.inverse_transform(true.reshape(-1,1)).reshape(batch_y.shape[0],-1,1)
                online_preds.append(pred)
                self.online_trues.append(true)

        # preds = np.array(preds)
        # trues = np.array(trues)
        # preds = np.concatenate(preds, axis=0)
        # trues = np.concatenate(trues, axis=0)
        # return preds, trues

        # preds, trues = self.trainer.test(self.test_loader, rescale=rescale, online=online, model=model)
        # return preds, trues
    
    def global_local_online_update(self, timestep, decay_update=False, adaptive=True):
        batch = get_tensor_batch(self.test_loader.dataset, timestep)
        inc_global_pred = self.incremental_model_ensemble(batch, decay_update=decay_update, type='global')
        inc_local_pred = self.incremental_model_ensemble(batch, decay_update=decay_update, type='local')
        local_pred, true = self.local_trainer.online_step(batch, return_preds=True)
        global_pred, _ = self.global_trainer.online_step(batch, return_preds=True)
        if adaptive:
            if self.args.consider_error:
                ensemble_pred, long_w, short_w, w = self.ensemble_trainer.online_step(batch, inc_global_pred, inc_local_pred, self.online_global_local_error_ratio)
                global_error_mean = np.mean(np.abs(inc_global_pred.detach().cpu().numpy() - true))
                local_error_mean = np.mean(np.abs(inc_local_pred.detach().cpu().numpy() - true))
                gloval_error_std = np.std(inc_global_pred.detach().cpu().numpy() - true)
                local_error_std = np.std(inc_local_pred.detach().cpu().numpy() - true)
                local_score = 1 / (local_error_mean + local_error_std)
                global_score = 1 / (global_error_mean + gloval_error_std)
                self.online_global_local_error_ratio = global_score / (local_score + global_score)
            else:
                ensemble_pred, long_w, short_w, w = self.ensemble_trainer.online_step(batch, inc_global_pred, inc_local_pred)
            self.online_long_weight.append(long_w.detach().cpu().numpy())
            self.online_short_weight.append(short_w.detach().cpu().numpy())
            self.online_weight.append(w.detach().cpu().numpy().reshape(-1))
            self.online_ensemble_preds.append(ensemble_pred.detach().cpu().numpy())
        self.online_global_preds.append(global_pred)
        self.online_local_preds.append(local_pred)
        self.online_inc_global_preds.append(inc_global_pred.detach().cpu().numpy())
        self.online_inc_local_preds.append(inc_local_pred.detach().cpu().numpy())
        self.online_trues.append(true)
        
        self.local_model_buffer.append(copy.deepcopy(self.local_trainer.model.state_dict()))
        if len(self.local_model_buffer) > self.args.model_buffer_size:
            self.local_model_buffer.pop(0)
        self.global_model_buffer.append(copy.deepcopy(self.global_trainer.model.state_dict()))
        if len(self.global_model_buffer) > self.args.model_buffer_size:
            self.global_model_buffer.pop(0)

    def online_update(self, timestep, model=None, decay_update=False):
        batch = get_tensor_batch(self.test_loader.dataset, timestep)
        inc_ensemble_output = self.incremental_model_ensemble(batch, decay_update=decay_update)
        self.online_inc_preds.append(inc_ensemble_output.detach().cpu())
        pred, true = self.trainer.online_step(batch, model=model, return_preds=True)
        self.online_preds.append(pred)
        self.online_trues.append(true)
        self.model_buffer.append(copy.deepcopy(self.trainer.model.state_dict()))
        if len(self.model_buffer) > self.args.model_buffer_size:
            self.model_buffer.pop(0)
    
    def incremental_model_ensemble(self, batch, decay_update=False, type=None):
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        if type == 'local':
            model_buffer = self.local_model_buffer
            model = self.local_model
            decay_rate = self.local_decay_rate
            trainer = self.local_trainer
            online_decay_rate = self.online_local_decay_rates
            decay_lr = self.args.local_decay_lr

        elif type == 'global':
            model_buffer = self.global_model_buffer
            model = self.global_model
            decay_rate = self.global_decay_rate
            trainer = self.global_trainer
            online_decay_rate = self.online_global_decay_rates
            decay_lr = self.args.global_decay_lr

        if len(model_buffer) == 0:
            model_buffer.append(model.state_dict())
        
        num_models = len(model_buffer)
        
        weights = torch.zeros(num_models, device=self.device)
    
        for i in range(num_models):
            weights[i] = torch.exp(-decay_rate * (num_models - i - 1)) 

        weights /= weights.sum()

        ensemble_output = torch.zeros(batch_y[:, -self.args.pred_len:, -1].shape, device=self.device)

        for i, model_state in enumerate(model_buffer):
            model_i = copy.deepcopy(model)
            model_i.load_state_dict(model_state)
            # model_i.eval()
            
            pred = trainer.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark, model_i)
            ensemble_output += weights[i] * pred 

        if decay_update:
            criterion = nn.MSELoss()
            batch_y = batch_y[:, -self.args.pred_len:, -1].float().to(self.device)
            loss = criterion(ensemble_output, batch_y)  
            loss.backward() 
            
            with torch.no_grad():
                decay_rate -= decay_lr * decay_rate.grad  
                decay_rate.grad.zero_()  
                online_decay_rate.append(decay_rate.detach().cpu().numpy())
        return ensemble_output
            
