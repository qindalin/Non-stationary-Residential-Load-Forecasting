import torch
from torch import nn
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class PatchTST(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2211.14730.pdf
    """

    def __init__(self, configs, patch_len=16, stride=8):
        """
        patch_len: int, patch len for patch_embedding
        stride: int, stride for patch_embedding
        """
        super().__init__()
        patch_len = getattr(configs, "patch_len", patch_len)
        stride = getattr(configs, "stride", stride)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        padding = stride

        # patching and embedding
        self.patch_embedding = PatchEmbedding(
            configs.d_model, patch_len, stride, padding, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )

        # Prediction Head
        self.head_nf = configs.d_model * \
                       int((configs.seq_len - patch_len) / stride + 2)

        self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                head_dropout=configs.dropout)
        adapter_hidden = max(16, min(64, configs.d_model // 8))
        seq_adapter_hidden = max(32, min(128, configs.d_model // 4))
        future_exogenous_dims = max(int(configs.enc_in) - 1, 1)
        self.future_exogenous_adapter = nn.Sequential(
            nn.Linear(2, adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, 1),
        )
        self.future_exogenous_sequence = nn.Sequential(
            nn.Linear(self.pred_len * future_exogenous_dims, seq_adapter_hidden),
            nn.GELU(),
            nn.Linear(seq_adapter_hidden, self.pred_len),
        )
        self.future_energy_head = nn.Sequential(
            nn.Linear(self.pred_len * future_exogenous_dims, seq_adapter_hidden),
            nn.GELU(),
            nn.Linear(seq_adapter_hidden, 1),
        )
        nn.init.zeros_(self.future_exogenous_adapter[-1].weight)
        nn.init.zeros_(self.future_exogenous_adapter[-1].bias)
        nn.init.zeros_(self.future_exogenous_sequence[-1].weight)
        nn.init.zeros_(self.future_exogenous_sequence[-1].bias)
        nn.init.zeros_(self.future_energy_head[-1].weight)
        nn.init.zeros_(self.future_energy_head[-1].bias)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Instance-wise normalization stabilizes PatchTST under client-specific scale shifts.
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        # do patching and embedding
        x_enc = x_enc.permute(0, 2, 1)
        # u: [bs * nvars x patch_num x d_model]
        enc_out, n_vars = self.patch_embedding(x_enc)

        # Encoder
        # z: [bs * nvars x patch_num x d_model]
        enc_out, attns = self.encoder(enc_out)
        # z: [bs x nvars x patch_num x d_model]
        enc_out = torch.reshape(
            enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        # z: [bs x nvars x d_model x patch_num]
        enc_out = enc_out.permute(0, 1, 3, 2)

        # Decoder
        dec_out = self.head(enc_out)  # z: [bs x nvars x target_window]
        dec_out = dec_out.permute(0, 2, 1)

        # Restore original scale after sequence normalization.
        dec_out = dec_out * stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        dec_out = dec_out + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, future_exogenous=None, x_dec=None, x_mark_dec=None, mask=None):
        if x_enc.dim() == 2:
            x_enc = x_enc.unsqueeze(-1)
        dec_out = self.forecast(x_enc, None, x_dec, x_mark_dec)
        load_pred = dec_out[:, -self.pred_len:, -1]
        if future_exogenous is not None:
            if future_exogenous.dim() == 2:
                future_exogenous = future_exogenous.unsqueeze(-1)
            if future_exogenous.shape[-1] > 0:
                exogenous_signal = future_exogenous[..., :1]
                adapter_input = torch.cat([load_pred.unsqueeze(-1), exogenous_signal], dim=-1)
                step_correction = self.future_exogenous_adapter(adapter_input).squeeze(-1)
                flat_exogenous = future_exogenous.reshape(future_exogenous.shape[0], -1)
                seq_correction = self.future_exogenous_sequence(flat_exogenous)
                energy_bias = self.future_energy_head(flat_exogenous).repeat(1, self.pred_len) / max(self.pred_len, 1)
                load_pred = load_pred + step_correction + seq_correction + energy_bias
        return load_pred
