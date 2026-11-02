import os
from torch import nn
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from utils.utils import pkl_save
from utils.tools import EarlyStopping, adjust_learning_rate
from tqdm import tqdm
from torch import optim
import torch.nn.functional as F

class Trainer:

    def __init__(self,
                 model: nn.Module,
                 args,
                 device,
                 path,
                 scaler,
                 model_name) -> None:
        self.args = args
        self.model_name = model_name
        self.model = model
        self.criterion = self._select_criterion()
        self.device = device
        self.scaler = scaler
        self.path = os.path.join(path, 'checkpoints/')
        os.makedirs(self.path, exist_ok=True)
        self.early_stopper = EarlyStopping(patience=self.args.patience, verbose=True) if self.args.patience > 0 else None

    def _select_optimizer(self, online=False):
        if online:
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.online_lr)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.lr)
        return model_optim
    
    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _energy_loss(self, outputs, target):
        weight = float(getattr(self.args, 'energy_loss_weight', 0.0) or 0.0)
        if weight <= 0.0:
            return torch.zeros((), device=outputs.device, dtype=outputs.dtype)
        pred_energy = outputs.sum(dim=-1)
        true_energy = target.sum(dim=-1)
        if getattr(self.args, 'energy_loss_relative', False):
            denom = torch.clamp(true_energy.abs(), min=1e-6)
            return weight * torch.mean(((pred_energy - true_energy) / denom) ** 2)
        return weight * torch.mean((pred_energy - true_energy) ** 2)
    
    def train(
            self,
            train_loader: DataLoader,
            vali_loader: DataLoader,
            epochs: int = None,
            save=False):
        E = epochs if epochs is not None else self.args.train_epochs
        optimizer = self._select_optimizer()
        
        if self.args.fed_type == 'fedprox':
            global_params = {k: v.clone().detach() for k, v in self.model.named_parameters()}

        for epoch in range(E):
            iter_count = 0
            train_loss = []
            self.model.train()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                optimizer.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_y = batch_y[:, -self.args.pred_len:, -1].to(self.device)
                loss = self.criterion(outputs, batch_y)
                loss = loss + self._energy_loss(outputs, batch_y)
                train_loss.append(loss.item())
                if self.args.fed_type == 'fedprox':
                    fedprox_loss = 0
                    for name, param in self.model.named_parameters():
                        if param.requires_grad:
                            fedprox_loss += torch.norm(param - global_params[name]) ** 2
                    # print(f'Fedprox loss: {fedprox_loss.item()}')
                    loss = loss + self.args.fedprox_reg * fedprox_loss

                loss.backward()
                optimizer.step()
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader)
            self.early_stopper(vali_loss, self.model, self.path)
            if self.early_stopper.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(optimizer, epoch + 1, self.args)
        
        best_model_path = self.path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if save:
            pkl_save(f'{self.path}/model.pkl', self.model)
        return self.model

    def vali(self, vali_loader, return_preds=False, rescale=False):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(vali_loader)):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
                batch_y = batch_y[:, -self.args.pred_len:, -1].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                loss = self.criterion(pred, true) + self._energy_loss(pred, true)
                total_loss.append(float(loss.item()))
                if rescale:
                    pred = self.scaler.inverse_transform(pred.reshape(-1,1)).reshape(batch_y.shape[0],-1,1)
                    true = self.scaler.inverse_transform(true.reshape(-1,1)).reshape(batch_y.shape[0],-1,1)
                preds.append(pred)
                trues.append(true)

        total_loss = np.average(total_loss)
        self.model.train()
        if return_preds:
            preds = np.array(preds)
            trues = np.array(trues)
            preds = np.concatenate(preds, axis=0)
            trues = np.concatenate(trues, axis=0)
            return preds, trues
        else:
            return total_loss


    def backbone_forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark, model=None):
        if model is None:
            model = self.model
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        if self.model_name in ['MLP', 'LSTM', 'CNN']:
            temp_input = batch_y[:, -self.args.pred_len:, 0].float()
            load_hist = batch_x[:,:,-1]
            outputs = model(load_hist, temp_input)
        elif self.model_name == 'PatchTST':
            future_exogenous = None
            if batch_y.shape[-1] > 1:
                future_exogenous = batch_y[:, -self.args.pred_len:, :-1].float()
            outputs = model(batch_x, future_exogenous)
        else:
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_label = batch_x[:, -self.args.label_len:, :]
            dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -self.args.pred_len:, -1]
        return outputs


    def online_step(self, batch, return_preds=False):
        self.model.train()
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        optimizer = self._select_optimizer(online=True)
        optimizer.zero_grad()
        
        outputs = self.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark)
        target = batch_y[:, -self.args.pred_len:, -1]
        loss = self.criterion(outputs, target) + self._energy_loss(outputs, target)
        loss.backward()
        optimizer.step()
        if return_preds:
            pred = outputs.detach().cpu().numpy() 
            true = target.detach().cpu().numpy() 
            return pred, true
        else:
            return outputs
    
class Ensemble_Trainer():
    def __init__(self, ensembler, long_weight, short_weight, args, device, scaler, model_name):
        self.ensembler = ensembler
        self.long_weight = long_weight
        self.short_weight = short_weight
        self.args = args
        self.device = device
        self.scaler = scaler
        self.model_name = model_name
        self.short_only = self.args.short_only
        self.long_only = self.args.long_only
        self.error_only = self.args.error_only
        self.criterion = self._select_criterion()

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion
    
    def train(self,
              train_loader: DataLoader,
              local_model: nn.Module,
              global_model: nn.Module,
              epochs: int = None,):
        E = epochs if epochs is not None else self.args.ensemble_epochs
        self.opt_w = optim.Adam([self.long_weight], lr=self.args.long_w_lr)
        self.opt_en = optim.Adam(self.ensembler.parameters(), lr=self.args.short_w_lr)

        if self.short_only:
              for epoch in range(E):
                print(f'Ensemble training [Epoch {epoch+1}/{E}]')
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    batch_x = batch_x.float().to(self.device) 
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y = batch_y.float()
                    long_w = F.sigmoid(self.long_weight)  
                    outputs, global_pred, local_pred = self.ensemble_forward(batch_x, 
                                                                            batch_y, 
                                                                            batch_x_mark, 
                                                                            batch_y_mark, 
                                                                            local_model, 
                                                                            global_model, 
                                                                            long_w)
                    batch_y = batch_y[:,-self.args.pred_len:,-1].to(self.device)
                    y1_w, y2_w = global_pred.detach(), local_pred.detach()
                    true_w = batch_y.detach()
                    inputs_ensemble = torch.cat([y1_w, y2_w, true_w], dim=1)
                    short_weight = self.ensembler(inputs_ensemble)
                    combine_w = F.sigmoid(short_weight)
                    
                    loss_ensemble = self.criterion(combine_w * y1_w + (1-combine_w) * y2_w, true_w)
                    loss_ensemble.backward()
                    self.opt_en.step()   
                    self.opt_en.zero_grad()  

        else:
            for epoch in range(E):
                print(f'Ensemble training [Epoch {epoch+1}/{E}]')
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                    batch_x = batch_x.float().to(self.device) 
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y = batch_y.float()
                    long_w = F.sigmoid(self.long_weight)  
                    outputs, global_pred, local_pred = self.ensemble_forward(batch_x, 
                                                                            batch_y, 
                                                                            batch_x_mark, 
                                                                            batch_y_mark, 
                                                                            local_model, 
                                                                            global_model, 
                                                                            long_w)
                    batch_y = batch_y[:,-self.args.pred_len:,-1].to(self.device)
                    loss_w = self.criterion(outputs, batch_y)
                    loss_w.backward()
                    self.opt_w.step()   
                    self.opt_w.zero_grad()   
                    
                    y1_w, y2_w = global_pred.detach(), local_pred.detach()
                    true_w = batch_y.detach()
                    long_w = F.sigmoid(self.long_weight)  
                    inputs_ensemble = torch.cat([long_w*y1_w, (1-long_w)*y2_w, true_w], dim=1)
                    short_weight = self.ensembler(inputs_ensemble)
                    combine_w = F.sigmoid(self.long_weight + short_weight)
                    
                    loss_ensemble = self.criterion(combine_w * y1_w + (1-combine_w) * y2_w, true_w)
                    loss_ensemble.backward()
                    self.opt_en.step()   
                    self.opt_en.zero_grad() 
             

    def online_step(self, batch, global_pred, local_pred, error_ratio=None):
        self.opt_w = optim.Adam([self.long_weight], lr=self.args.long_w_lr)
        self.opt_en = optim.Adam(self.ensembler.parameters(), lr=self.args.short_w_lr)
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch
        batch_y = batch_y[:,-self.args.pred_len:,-1].float().to(self.device)

        true = batch_y.detach()
        y1, y2 = global_pred.detach(), local_pred.detach()
        error_bias = None
        if error_ratio is not None:
            error_bias = torch.as_tensor(error_ratio, device=self.device, dtype=y1.dtype)

        if self.long_only:
            combine_w = F.sigmoid(self.long_weight)
            outputs = combine_w * y1 + (1-combine_w) * y2
            loss_long_w = self.criterion(outputs, batch_y)
            loss_long_w.backward()
            self.opt_w.step()
            self.opt_w.zero_grad()
            combine_w = F.sigmoid(self.long_weight)
            outputs = combine_w * y1 + (1-combine_w) * y2
            weight = combine_w

        elif self.short_only:
            inputs_ensemble = torch.cat([y1, y2, true], dim=1)
            self.short_weight = self.ensembler(inputs_ensemble)
            combine_w = F.sigmoid(self.short_weight)
            loss_short_w = self.criterion(combine_w * y1 + (1-combine_w) * y2, true)
            loss_short_w.backward(retain_graph=True)
            self.opt_en.step()
            self.opt_en.zero_grad()
            self.short_weight = self.ensembler(inputs_ensemble, train=False)
            combine_w = F.sigmoid(self.short_weight)
            outputs = combine_w * y1 + (1-combine_w) * y2
            weight = combine_w

        elif self.error_only:
            weight = torch.as_tensor(error_ratio, device=self.device, dtype=y1.dtype)
            outputs = weight * y1 + (1-weight) * y2

        else:
            long_w = F.sigmoid(self.long_weight) 
            inputs_ensemble = torch.cat([long_w*y1, (1-long_w)*y2, true], dim=1)
            self.short_weight = self.ensembler(inputs_ensemble)
            if error_bias is not None:
                combine_w = F.sigmoid(self.long_weight + self.short_weight + error_bias)
            else:
                combine_w = F.sigmoid(self.long_weight + self.short_weight)
            loss_short_w = self.criterion(combine_w * y1 + (1-combine_w) * y2, true)
            loss_short_w.backward(retain_graph=True)
            self.opt_en.step()
            self.opt_en.zero_grad()

            long_w = F.sigmoid(self.long_weight) 
            loss_long_w = self.criterion(long_w * global_pred.detach() + (1-long_w) * local_pred.detach(), batch_y)
            loss_long_w.backward()
            self.opt_w.step()
            self.opt_w.zero_grad()
            long_w = F.sigmoid(self.long_weight)
            inputs_ensemble = torch.cat([long_w * y1, (1 - long_w) * y2, true], dim=1)
            self.short_weight = self.ensembler(inputs_ensemble, train=False)
            if error_bias is not None:
                combine_w = F.sigmoid(self.long_weight + self.short_weight + error_bias)
            else:
                combine_w = F.sigmoid(self.long_weight + self.short_weight)
            outputs = combine_w * y1 + (1 - combine_w) * y2
            weight = combine_w
        return outputs, self.long_weight, self.short_weight, weight

    def backbone_forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark, model):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        if self.model_name in ['MLP', 'LSTM', 'CNN']:
            temp_input = batch_y[:, -self.args.pred_len:, 0].float()
            load_hist = batch_x[:,:,-1]
            outputs = model(load_hist, temp_input)
        elif self.model_name == 'PatchTST':
            future_exogenous = None
            if batch_y.shape[-1] > 1:
                future_exogenous = batch_y[:, -self.args.pred_len:, :-1].float()
            outputs = model(batch_x, future_exogenous)
        else:
            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_label = batch_x[:, -self.args.label_len:, :]
            dec_inp = torch.cat([dec_label, dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            outputs = outputs[:, -self.args.pred_len:, -1]
        return outputs
    
    def ensemble_forward(self, batch_x, batch_y, batch_x_mark, batch_y_mark, local_model, global_model, weight):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)
        global_pred = self.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark, global_model)
        local_pred = self.backbone_forward(batch_x, batch_y, batch_x_mark, batch_y_mark, local_model)
        outputs = weight * global_pred + (1-weight) * local_pred
        return outputs, global_pred, local_pred
