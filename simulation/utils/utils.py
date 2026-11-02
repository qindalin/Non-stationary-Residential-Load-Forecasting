import os
import numpy as np
import pickle
import torch
import random
from datetime import datetime
import torch.nn as nn
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

class arg():
    def __init__(
            self,
            batch_size = 32,
            lr = 0.0001,
            online_lr = 0.001,
            global_decay_lr = 0.1,
            local_decay_lr = 0.1,
            long_w_lr = 0.5,
            short_w_lr = 0.0001,
            fedprox_reg = 0.01,
            fedsgd_lr = 1,
            fedavgm_lr = 0.1,
            fedavgm_beta = 0.9,
            short_only = False,
            long_only = False,
            error_only = False,
            consider_error = False,
            decay_rate = 0.0,
            repr_dims = 320,
            iters = None,
            alpha = 0.0005,
            kernels = [1, 2, 4, 8, 16, 32, 64, 128],
            gpu = 0,
            seed = 0,
            max_threads = 8,
            hidden_dims = 64,
            pred_len = 24,
            label_len = 0,
            seq_len = 96,
            enc_in = 1,
            output_dims = 320,
            model_buffer_size = 7,
            device = 'cuda',
            moving_avg = 25,
            freq = 'h',
            data = 'custom',
            features = 'MS',
            seasonal_patterns = 'Monthly',
            target = 'value',
            num_workers = 10,
            embed = 'timeF',
            root_path = 'data',
            patience = 3,
            train_epochs = 10,
            ensemble_epochs = 10,
            fed_type = None,
            fed_rounds = 10,
            fed_local_epochs = 1,
            lradj = 'type1',
            inverse = False,
            drop_out = 0.1,
            d_ff = 2048,
            d_model = 512,
            n_heads = 8,
            e_layers = 2,
            d_layers = 1,
            patch_len = 16,
            stride = 8,
            activation = 'gelu',
            factor = 1,
            train_ratio = None,
            energy_loss_weight = 0.0,
            energy_loss_relative = False,
    ):
        self.batch_size = batch_size
        self.fed_type = fed_type
        self.fedprox_reg = fedprox_reg
        self.fedavgm_beta = fedavgm_beta
        self.fedsgd_lr = fedsgd_lr
        self.fedavgm_lr = fedavgm_lr
        self.fed_rounds = fed_rounds
        self.fed_local_epochs = fed_local_epochs
        self.lr = lr
        self.online_lr = online_lr
        self.global_decay_lr = global_decay_lr
        self.local_decay_lr = local_decay_lr
        self.long_w_lr = long_w_lr
        self.short_w_lr = short_w_lr
        self.short_only = short_only
        self.long_only = long_only
        self.error_only = error_only
        self.consider_error = consider_error
        self.decay_rate = decay_rate
        self.repr_dims = repr_dims
        self.iters = iters
        self.alpha = alpha
        self.kernels = kernels
        self.gpu = gpu
        self.seed = seed
        self.max_threads = max_threads
        self.hidden_dims = hidden_dims
        self.pred_len = pred_len
        self.label_len =label_len
        self.seq_len = seq_len
        self.enc_in = enc_in
        self.output_dims = output_dims
        self.model_buffer_size = model_buffer_size
        self.device = device
        self.moving_avg = moving_avg
        self.freq = freq
        self.data = data
        self.features = features
        self.seasonal_patterns = seasonal_patterns
        self.target = target
        self.num_workers = num_workers
        self.embed = embed
        self.root_path = root_path
        self.patience = patience
        self.train_epochs = train_epochs
        self.ensemble_epochs = ensemble_epochs
        self.lradj = lradj
        self.inverse = inverse
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.patch_len = patch_len
        self.stride = stride
        self.d_ff = d_ff
        self.dropout = drop_out
        self.activation = activation
        self.factor = factor
        self.train_ratio = train_ratio
        self.energy_loss_weight = energy_loss_weight
        self.energy_loss_relative = energy_loss_relative

def cumavg(m):
    cumsum= np.cumsum(m)
    return cumsum / np.arange(1, cumsum.size + 1)


def safe_smape(true, pred, eps=1e-6):
    true = np.asarray(true, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    denom = np.maximum(np.abs(true) + np.abs(pred), eps)
    return float(np.mean(2.0 * np.abs(true - pred) / denom) * 100.0)


def safe_epe(true, pred, eps=1e-6):
    true = np.asarray(true, dtype=np.float64).reshape(-1)
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)
    denom = max(float(np.sum(true)), eps)
    return float(abs(float(np.sum(pred)) - float(np.sum(true))) / denom)


def inverse_with_scaler(array, scaler):
    arr = np.asarray(array)
    flat = arr.reshape(-1, 1)
    if hasattr(scaler, "scaler") and hasattr(scaler.scaler, "scale_"):
        sk = scaler.scaler
    else:
        sk = scaler
    scale = np.asarray(getattr(sk, "scale_", [1.0]))
    mean = np.asarray(getattr(sk, "mean_", [0.0]))
    if scale.size == 1:
        restored = flat * scale[0] + mean[0]
    else:
        restored = flat * scale[-1] + mean[-1]
    return restored.reshape(arr.shape)

def pkl_save(name, var):
    with open(name, 'wb') as f:
        pickle.dump(var, f)

def pkl_load(name):
    with open(name, 'rb') as f:
        return pickle.load(f)
    
def torch_pad_nan(arr, left=0, right=0, dim=0):
    if left > 0:
        padshape = list(arr.shape)
        padshape[dim] = left
        arr = torch.cat((torch.full(padshape, np.nan), arr), dim=dim)
    if right > 0:
        padshape = list(arr.shape)
        padshape[dim] = right
        arr = torch.cat((arr, torch.full(padshape, np.nan)), dim=dim)
    return arr
    
def pad_nan_to_target(array, target_length, axis=0, both_side=False):
    assert array.dtype in [np.float16, np.float32, np.float64]
    pad_size = target_length - array.shape[axis]
    if pad_size <= 0:
        return array
    npad = [(0, 0)] * array.ndim
    if both_side:
        npad[axis] = (pad_size // 2, pad_size - pad_size//2)
    else:
        npad[axis] = (0, pad_size)
    return np.pad(array, pad_width=npad, mode='constant', constant_values=np.nan)

def split_with_nan(x, sections, axis=0):
    assert x.dtype in [np.float16, np.float32, np.float64]
    arrs = np.array_split(x, sections, axis=axis)
    target_length = arrs[0].shape[axis]
    for i in range(len(arrs)):
        arrs[i] = pad_nan_to_target(arrs[i], target_length, axis=axis)
    return arrs

def take_per_row(A, indx, num_elem):
    all_indx = indx[:,None] + np.arange(num_elem)
    return A[torch.arange(all_indx.shape[0])[:,None], all_indx]

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]

def data_dropout(arr, p):
    B, T = arr.shape[0], arr.shape[1]
    mask = np.full(B*T, False, dtype=np.bool)
    ele_sel = np.random.choice(
        B*T,
        size=int(B*T*p),
        replace=False
    )
    mask[ele_sel] = True
    res = arr.copy()
    res[mask.reshape(B, T)] = np.nan
    return res

def name_with_datetime(prefix='default'):
    now = datetime.now()
    return prefix + '_' + now.strftime("%Y%m%d_%H%M%S")

def init_dl_program(
    device_name,
    seed=None,
    use_cudnn=True,
    deterministic=False,
    benchmark=False,
    use_tf32=False,
    max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)
        
    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)
        
    if isinstance(device_name, (str, int)):
        device_name = [device_name]
    
    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    
    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32
        
    return devices if len(devices) > 1 else devices[0]

def get_tensor_batch(dataset, index):
    raw_batch = dataset[index]
    return (
        torch.from_numpy(raw_batch[0]).unsqueeze(0),
        torch.from_numpy(raw_batch[1]).unsqueeze(0),
        torch.from_numpy(raw_batch[2]).unsqueeze(0),
        torch.from_numpy(raw_batch[3]).unsqueeze(0)
    )

def sliding_window_metrics(true, pred, window_size):
    mse_list = []
    mape_list = []
    mae_list = []

    for i in range(len(true) - window_size + 1):
        true_window = true[i:i + window_size]
        pred_window = pred[i:i + window_size]

        mse = mean_squared_error(true_window, pred_window)
        mape = mean_absolute_percentage_error(true_window, pred_window)
        mae = mean_absolute_error(true_window, pred_window)

        mse_list.append(mse)
        mape_list.append(mape)
        mae_list.append(mae)
    return mse_list, mape_list, mae_list

def save_result(results, save_path, save_name):
    results_df = pd.DataFrame(results).set_index('Client')
    os.makedirs(save_path, exist_ok=True)
    results_df.to_csv(os.path.join(save_path, f"{save_name}_result.csv"))

def cumavg(m):
    cumsum= np.cumsum(m)
    return cumsum / np.arange(1, cumsum.size + 1) 


def plot_cum_result(ground_truth, local_ol_preds, fed_ol_preds, ensemble_ol_preds, server, metric='MSE'):
    if metric == 'MSE':
        metric_func = mean_squared_error
    elif metric == 'MAE':
        metric_func = mean_absolute_error

    # 字典存储误差分组 (按 client_type)
    local_ol_error_by_type = defaultdict(list)
    fed_ol_error_by_type = defaultdict(list)
    ensemble_ol_error_by_type = defaultdict(list)

    # 遍历客户数据并计算误差
    for client, true, local_ol_pred, fed_ol_pred, ensemble_pred in zip(server.clients, ground_truth, local_ol_preds, fed_ol_preds, ensemble_ol_preds):
        name = client.data_name
        _, client_type, _ = name.split("_")  # 提取客户类型

        # 每个客户的误差列表
        local_error = []
        fed_error = []
        ensemble_error = []

        # 将数据 reshape 后按天计算误差
        true = true.reshape(-1, 24)
        local_ol_pred = local_ol_pred.reshape(-1, 24)
        fed_ol_pred = fed_ol_pred.reshape(-1, 24)
        ensemble_pred = ensemble_pred.reshape(-1, 24)

        for i in range(true.shape[0]):  # 按天计算误差
            local_error.append(metric_func(true[i], local_ol_pred[i]))
            fed_error.append(metric_func(true[i], fed_ol_pred[i]))
            ensemble_error.append(metric_func(true[i], ensemble_pred[i]))

        # 将误差累计到对应 client_type 中
        local_ol_error_by_type[client_type].append(local_error)
        fed_ol_error_by_type[client_type].append(fed_error)
        ensemble_ol_error_by_type[client_type].append(ensemble_error)

    # 按客户类型计算均值并绘制图
    fig, ax = plt.subplots(len(local_ol_error_by_type), 1, figsize=(8, 6 * len(local_ol_error_by_type)))

    if len(local_ol_error_by_type) == 1:  # 处理只有一种类型的情况
        ax = [ax]

    for idx, client_type in enumerate(local_ol_error_by_type.keys()):
        # 计算均值误差
        local_mean_error = np.mean(np.array(local_ol_error_by_type[client_type]), axis=0)
        fed_mean_error = np.mean(np.array(fed_ol_error_by_type[client_type]), axis=0)
        ensemble_mean_error = np.mean(np.array(ensemble_ol_error_by_type[client_type]), axis=0)

        # 计算累计均值误差
        local_cum_error = cumavg(local_mean_error)
        fed_cum_error = cumavg(fed_mean_error)
        ensemble_cum_error = cumavg(ensemble_mean_error)

        # 绘制图
        ax[idx].plot(local_cum_error, linestyle='--', label='Local Online')
        ax[idx].plot(fed_cum_error, linestyle='--', label='Fed Online')
        ax[idx].plot(ensemble_cum_error, linewidth=1.5, label='Proposed')
        ax[idx].set_title(f'Cumulative {metric} for Client Type: {client_type}', size=15)
        ax[idx].set_xlabel('Day', size=15)
        ax[idx].set_ylabel(f'Cumulative {metric}', size=15)
        ax[idx].tick_params(labelsize=15)
        ax[idx].legend(prop={'size': 15})

    plt.tight_layout()
    plt.show()
