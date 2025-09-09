import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from local_attention.rotary import SinusoidalEmbeddings, apply_rotary_pos_emb
from local_attention import LocalAttention

class MDM(nn.Module):
    def __init__(self, modeltype, njoints, nfeats,
                 latent_dim=256, ff_size=1024, num_layers=8, num_heads=4, dropout=0.1,
                 ablation=None, activation="gelu", legacy=False, data_rep='rot6d', dataset='amass', clip_dim=512,
                 arch='trans_enc', emb_trans_dec=False, audio_feat='', n_seed=1, cond_mode='', device='cpu',
                 style_dim=-1, source_audio_dim=-1, audio_feat_dim_latent=-1, model_name='Ref-Basic', **kargs):
        super().__init__()

        self.legacy = legacy
        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.data_rep = data_rep
        self.dataset = dataset
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.clip_dim = clip_dim
        self.input_feats = self.njoints * self.nfeats
        self.normalize_output = kargs.get('normalize_encoder_output', False)
        self.cond_mask_prob = kargs.get('cond_mask_prob', 0.)
        self.arch = arch
        self.gru_emb_dim = self.latent_dim if self.arch == 'gru' else 0
        self.source_audio_dim = source_audio_dim
        self.audio_feat = audio_feat
        self.model_name = model_name

        # --- Selección de Arquitectura y Dimensiones ---
        if self.model_name == 'Ref-Basic':
            print('USE Ref-Basic MODEL with Simple WavEncoder')
            self.audio_feat_dim = 512 # Dimensión correcta para el checkpoint
            self.WavEncoder = WavEncoderSimple(self.source_audio_dim, self.audio_feat_dim)
        elif self.model_name == 'Basic-Whisper':
            print('USE Basic-Whisper MODEL')
            self.audio_feat_dim = 512 # Dimensión correcta para el checkpoint
            self.WavEncoder = WavEncoderSimple(self.source_audio_dim, self.audio_feat_dim)
        elif self.model_name == 'Multi-Fusion':
            print('USE Multi-Fusion MODEL with Standard WavEncoder')
            self.audio_feat_dim = audio_feat_dim_latent
            self.WavEncoder = WavEncoder(self.source_audio_dim, self.audio_feat_dim)
        elif self.model_name == 'Multi-Dual':
            print('USE Multi-Dual MODEL')
            self.audio_feat_dim = audio_feat_dim_latent
            self.audio_encoder = nn.Sequential(
                nn.Linear(1133, 512), nn.ReLU(),
                nn.Linear(512, self.audio_feat_dim)
            )
            self.text_encoder = nn.Sequential(
                nn.Linear(3074, 1500), nn.ReLU(),
                nn.Linear(1500, 512), nn.ReLU(),
                nn.Linear(512, self.audio_feat_dim)
            )
        elif self.model_name == 'Text-Only':
            print('USE Text-Only MODEL')
            self.text_feat_dim = 256
            self.text_encoder = nn.Sequential(
                nn.Linear(3074, 1500), nn.ReLU(),
                nn.Linear(1500, 512), nn.ReLU(),
                nn.Linear(512, self.text_feat_dim)
            )
        elif self.model_name == 'Multi-Whisper':
            print('USE Multi-Whisper MODEL')
            self.audio_feat_dim = 512 # Dimensión correcta para el checkpoint
            self.audio_encoder = nn.Sequential(
                nn.Linear(1281, 512), nn.ReLU(),
                nn.Linear(512, self.audio_feat_dim)
            )
            self.text_encoder = nn.Sequential(
                nn.Linear(3074, 1500), nn.ReLU(),
                nn.Linear(1500, 512), nn.ReLU(),
                nn.Linear(512, self.audio_feat_dim)
            )
        elif self.model_name == 'Multi-DiT':
            print('USE Multi-DiT MODEL with DiT WavEncoder')
            self.audio_feat_dim = 512
            self.WavEncoder = WavEncoderDiT(self.source_audio_dim, self.audio_feat_dim)
        elif self.model_name == 'Text-DiT':
            print('USE Text-DiT MODEL')
            self.text_feat_dim = 256
            self.TextEncoder = TextEncoder(3074, self.text_feat_dim)
            self.text_projector = nn.Linear(self.text_feat_dim, self.latent_dim)

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        self.emb_trans_dec = emb_trans_dec
        self.cond_mode = cond_mode
        self.num_head = 8
        self.input_process = InputProcess(self.data_rep, self.input_feats + self.gru_emb_dim, self.latent_dim)

        if self.arch == 'trans_enc':
            if self.model_name in ['Multi-DiT', 'Text-DiT']:
                print("TRANS_ENC init with Cross-Attention (DiT-style, AdaLN, Perceiver)")
                self.seqTransEncoder = DiTTransformer(
                    d_model=self.latent_dim, nhead=self.num_heads, num_layers=self.num_layers,
                    dim_feedforward=self.ff_size, dropout=self.dropout, activation=self.activation
                )
            else:
                print("TRANS_ENC init")
                seqTransEncoderLayer = nn.TransformerEncoderLayer(
                    d_model=self.latent_dim, nhead=self.num_heads, dim_feedforward=self.ff_size,
                    dropout=self.dropout, activation=self.activation
                )
                self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        else:
            raise ValueError('Please choose correct architecture [trans_enc]')

        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)
        self.n_seed = n_seed
        if 'style1' in self.cond_mode:
            print('EMBED STYLE BEGIN TOKEN')
            self.style_dim = self.latent_dim
            self.embed_style = nn.Linear(style_dim, self.style_dim)
            if self.model_name in ['Multi-Dual', 'Text-Only']:
                 self.embed_text = nn.Linear(self.njoints, 256)
            elif self.model_name == 'Multi-Whisper':
                self.embed_text = nn.Linear(self.njoints, 1024)
            elif self.model_name == 'Basic-Whisper':
                self.embed_text = nn.Linear(self.njoints, 512) # Dimensión correcta
            elif self.model_name in ['Multi-DiT', 'Text-DiT']:
                 self.embed_text = nn.Linear(self.njoints, self.latent_dim)
            else: # Ref-Basic, Multi-Fusion
                self.embed_text = nn.Linear(self.njoints, self.audio_feat_dim)

        self.output_process = OutputProcess(self.data_rep, self.input_feats, self.latent_dim, self.njoints, self.nfeats)

        if 'cross_local_attention' in self.cond_mode:
            self.rel_pos = SinusoidalEmbeddings(self.latent_dim // self.num_head)
            self.cross_local_attention = LocalAttention(
                dim=48, window_size=15, causal=True, look_backward=1,
                look_forward=0, dropout=0.1, exact_windowsize=False
            )
            if self.model_name in ['Multi-Dual', 'Text-Only']:
                self.input_process2 = nn.Linear(self.latent_dim + 256 + self.latent_dim, self.latent_dim)
            elif self.model_name == 'Multi-Whisper':
                 self.input_process2 = nn.Linear(self.latent_dim + 1024 + self.latent_dim, self.latent_dim)
            elif self.model_name == 'Basic-Whisper':
                 self.input_process2 = nn.Linear(self.latent_dim * 2 + 512, self.latent_dim) # Dimensión correcta
            elif self.model_name == 'Multi-DiT':
                self.input_process2 = nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim)
            elif self.model_name == 'Text-DiT':
                self.input_process2 = nn.Linear(self.latent_dim + self.latent_dim + self.latent_dim, self.latent_dim)
            else: # Ref-Basic, Multi-Fusion
                self.input_process2 = nn.Linear(self.latent_dim * 2 + self.audio_feat_dim, self.latent_dim)
    
    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, timesteps, y=None):
        bs, njoints, nfeats, nframes = x.shape
        emb_t = self.embed_timestep(timesteps)
        force_mask = y.get('uncond', False)
        embed_style = self.mask_cond(self.embed_style(y['style']), force_mask=force_mask)

        if 'cross_local_attention4' in self.cond_mode:
            if self.model_name in ['Ref-Basic', 'Basic-Whisper', 'Multi-Fusion']:
                embed_text = self.embed_text(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)
                enc_text = self.WavEncoder(y['audio']).permute(1, 0, 2)
                enc_text = torch.cat((embed_text, enc_text), axis=0)
            elif self.model_name in ['Multi-Dual', 'Multi-Whisper']:
                if self.model_name == 'Multi-Dual':
                    audio_input = y['audio'][..., :1133]
                    text_input = y['audio'][..., 1133:4207]
                else: # Multi-Whisper
                    audio_input = y['audio'][..., :1281]
                    text_input = y['audio'][..., 1281:4355]
                audio_rep = self.audio_encoder(audio_input)
                text_rep = self.text_encoder(text_input)
                enc_text = torch.cat((audio_rep, text_rep), dim=-1).permute(1, 0, 2)
                embed_text = self.embed_text(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)
                enc_text = torch.cat((embed_text, enc_text), axis=0)
            elif self.model_name == 'Text-Only':
                text_input = y['audio'][..., 1133:4207]
                enc_text = self.text_encoder(text_input).permute(1,0,2)
                embed_text = self.embed_text(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)
                enc_text = torch.cat((embed_text, enc_text), axis=0)
            elif self.model_name == 'Multi-DiT':
                enc_text = self.WavEncoder(y['audio']).permute(1, 0, 2)
                embed_text = self.embed_text(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)
                context = torch.cat((embed_text, enc_text), axis=0)
            elif self.model_name == 'Text-DiT':
                text_input = y['audio'][..., 1133:1133+3074]
                enc_text = self.TextEncoder(text_input)
                enc_text = self.text_projector(enc_text).permute(1, 0, 2)
                embed_text = self.embed_text(y['seed'].squeeze(2).permute(0, 2, 1)).permute(1, 0, 2)
                context = torch.cat((embed_text, enc_text), axis=0)


            x = x.reshape(bs, njoints * nfeats, 1, nframes)
            x_ = self.input_process(x)

            if self.model_name in ['Multi-DiT', 'Text-DiT']:
                 xseq = x_
                 embed_style_2 = (embed_style + emb_t).repeat(nframes, 1, 1)
                 if self.model_name == 'Multi-DiT':
                     xseq = torch.cat((embed_style_2, xseq), axis=2)
                 else:
                     xseq = torch.cat((embed_style_2, xseq, context), axis=2)
                 xseq = self.input_process2(xseq)
            else:
                packed_shape = [torch.Size([bs, self.num_head])]
                xseq = torch.cat((x_, enc_text), axis=2)
                embed_style_2 = (embed_style + emb_t).repeat(nframes, 1, 1)
                xseq = torch.cat((embed_style_2, xseq), axis=2)
                xseq = self.input_process2(xseq)


            xseq = xseq.permute(1, 0, 2)
            xseq = xseq.view(bs, nframes, self.num_head, -1)
            xseq = xseq.permute(0, 2, 1, 3)
            xseq = xseq.reshape(bs * self.num_head, nframes, -1)
            pos_emb = self.rel_pos(xseq)
            xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
            xseq = self.cross_local_attention(xseq, xseq, xseq, packed_shape=[torch.Size([bs, self.num_head])],
                                              mask=y['mask_local'])
            xseq = xseq.permute(0, 2, 1, 3)
            xseq = xseq.reshape(bs, nframes, -1)
            xseq = xseq.permute(1, 0, 2)
            xseq = torch.cat((embed_style + emb_t, xseq), axis=0)

            if self.model_name in ['Multi-DiT', 'Text-DiT']:
                context = torch.cat((embed_style + emb_t, context), axis=0)
                time_emb = (embed_style + emb_t).repeat(nframes + 1, 1, 1)
                output = self.seqTransEncoder(xseq, context, time_emb)[1:]
            else:
                xseq = xseq.permute(1, 0, 2)
                xseq = xseq.view(bs, nframes + 1, self.num_head, -1)
                xseq = xseq.permute(0, 2, 1, 3)
                xseq = xseq.reshape(bs * self.num_head, nframes + 1, -1)
                pos_emb = self.rel_pos(xseq)
                xseq, _ = apply_rotary_pos_emb(xseq, xseq, pos_emb)
                xseq_rpe = xseq.reshape(bs, self.num_head, nframes + 1, -1)
                xseq = xseq_rpe.permute(0, 2, 1, 3)
                xseq = xseq.view(bs, nframes + 1, -1)
                xseq = xseq.permute(1, 0, 2)
                output = self.seqTransEncoder(xseq)[1:]

        output = self.output_process(output)
        return output


class WavEncoder(nn.Module):
    """Codificador estándar con 3 capas lineales."""
    def __init__(self, source_dim, audio_feat_dim):
        super().__init__()
        self.audio_feature_map = nn.Sequential(
            nn.Linear(source_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, audio_feat_dim)
        )
    def forward(self, rep):
        return self.audio_feature_map(rep)

class WavEncoderSimple(nn.Module):
    """Codificador simple con 1 capa lineal para Ref-Basic."""
    def __init__(self, source_dim, audio_feat_dim):
        super().__init__()
        self.audio_feature_map = nn.Linear(source_dim, audio_feat_dim)
    def forward(self, rep):
        return self.audio_feature_map(rep)

class WavEncoderDiT(nn.Module):
    """Codificador con 2 capas lineales para Multi-DiT."""
    def __init__(self, source_dim, audio_feat_dim):
        super().__init__()
        self.audio_feature_map = nn.Sequential(
            nn.Linear(source_dim, 1024), nn.ReLU(),
            nn.Linear(1024, audio_feat_dim)
        )
    def forward(self, rep):
        return self.audio_feature_map(rep)



class PerceiverCrossAttention(nn.Module):
    def __init__(self, d_model, nhead, num_latents=32, dropout=0.1):
        super().__init__()
        self.num_latents = num_latents
        self.latent_tokens = nn.Parameter(torch.randn(num_latents, d_model))
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, x, context):
        bs = x.shape[1]
        latent = self.latent_tokens.unsqueeze(1).repeat(1, bs, 1)  # (num_latents, bs, d_model)
        latent = self.cross_attn(latent, context, context)[0]  # (32, 64, 256)
        x = self.cross_attn(x, latent, latent)[0]  # (151, 64, 512)
        return x

class DiTTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation="gelu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn = PerceiverCrossAttention(d_model, nhead, num_latents=32, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.scale_shift1 = nn.Linear(d_model, d_model * 2)
        self.scale_shift2 = nn.Linear(d_model, d_model * 2)
        self.scale_shift3 = nn.Linear(d_model, d_model * 2)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, context, time_emb):
        scale_shift = self.scale_shift1(time_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = self.norm1(x) * (1 + scale) + shift
        x2 = self.self_attn(x, x, x)[0]
        x = x + self.dropout1(x2)

        scale_shift = self.scale_shift2(time_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = self.norm2(x) * (1 + scale) + shift
        x2 = self.cross_attn(x, context)
        x = x + self.dropout2(x2)

        scale_shift = self.scale_shift3(time_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        x = self.norm3(x) * (1 + scale) + shift
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = x + self.dropout3(x2)

        return x

class DiTTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout, activation="gelu"):
        super().__init__()
        self.layers = nn.ModuleList([
            DiTTransformerLayer(d_model, nhead, dim_feedforward, dropout, activation)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, context, time_emb):
        for layer in self.layers:
            x = layer(x, context, time_emb)
        return self.norm(x)

class TextEncoder(nn.Module):
    def __init__(self, input_dim, text_feat_dim):
        super().__init__()
        self.text_feature_map = nn.Sequential(
            nn.Linear(input_dim, 1500),  # Reducción inicial a 1500
            nn.ReLU(),
            nn.Linear(1500, 512),       # Reducción a 512
            nn.ReLU(),
            nn.Linear(512, text_feat_dim)  # Final a 256
        )

    def forward(self, rep):
        rep = self.text_feature_map(rep)
        return rep


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)      # (5000, 128)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)     # (5000, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
        self, num_positions: int, embedding_dim: int
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, seq_len: int, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)


class InputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        if self.data_rep == 'rot_vel':
            self.velEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        bs, njoints, nfeats, nframes = x.shape
        x = x.permute((3, 0, 1, 2)).reshape(nframes, bs, njoints*nfeats)

        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            x = self.poseEmbedding(x)  # [seqlen, bs, d]
            return x
        elif self.data_rep == 'rot_vel':
            first_pose = x[[0]]  # [1, bs, 150]
            first_pose = self.poseEmbedding(first_pose)  # [1, bs, d]
            vel = x[1:]  # [seqlen-1, bs, 150]
            vel = self.velEmbedding(vel)  # [seqlen-1, bs, d]
            return torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, d]
        else:
            raise ValueError


class OutputProcess(nn.Module):
    def __init__(self, data_rep, input_feats, latent_dim, njoints, nfeats):
        super().__init__()
        self.data_rep = data_rep
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        if self.data_rep == 'rot_vel':
            self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output):
        nframes, bs, d = output.shape
        if self.data_rep in ['rot6d', 'xyz', 'hml_vec']:
            output = self.poseFinal(output)  # [seqlen, bs, 150]
        elif self.data_rep == 'rot_vel':
            first_pose = output[[0]]  # [1, bs, d]
            first_pose = self.poseFinal(first_pose)  # [1, bs, 150]
            vel = output[1:]  # [seqlen-1, bs, d]
            vel = self.velFinal(vel)  # [seqlen-1, bs, 150]
            output = torch.cat((first_pose, vel), axis=0)  # [seqlen, bs, 150]
        else:
            raise ValueError
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)  # [bs, njoints, nfeats, nframes]
        return output


class LinearTemporalCrossAttention(nn.Module):

    def __init__(self, seq_len, latent_dim, text_latent_dim, num_head, dropout, time_embed_dim):
        super().__init__()
        self.num_head = num_head
        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)
        self.query = nn.Linear(latent_dim, latent_dim)
        self.key = nn.Linear(text_latent_dim, latent_dim)
        self.value = nn.Linear(text_latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj_out = nn.Linear(latent_dim, latent_dim)

    def forward(self, x, xf=None, emb=None):
        """
        x: B, T, D      , [240, 2, 256]
        xf: B, N, L     , [1, 2, 256]
        """
        x = x.permute(1, 0, 2)
        # xf = xf.permute(1, 0, 2)
        B, T, D = x.shape
        # N = xf.shape[1]
        H = self.num_head
        # B, T, D
        query = self.query(self.norm(x))
        # B, N, D
        key = self.key(self.text_norm(x))
        query = F.softmax(query.view(B, T, H, -1), dim=-1)
        key = F.softmax(key.view(B, T, H, -1), dim=1)
        # B, N, H, HD
        value = self.value(self.text_norm(x)).view(B, T, H, -1)
        # B, H, HD, HD
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)
        y = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        # y = x + self.proj_out(y, emb)
        return y


if __name__ == '__main__':
    '''
    cd ./BEAT-main/model
    python mdm.py
    '''
    n_frames = 150
    n_seed = 30
    njoints = 684*3
    audio_feature_dim = 1133 + 301      # audio_f + text_f
    style_dim = 2
    bs = 2
    
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = MDM(modeltype='', njoints=njoints, nfeats=1, cond_mode='cross_local_attention5_style1', audio_feat='wavlm',
                arch='trans_enc', latent_dim=512, n_seed=n_seed, cond_mask_prob=0.1, 
                style_dim=style_dim, source_audio_dim=audio_feature_dim).to(device)

    x = torch.randn(bs, njoints, 1, n_frames)
    t = torch.tensor([12, 85])

    model_kwargs_ = {'y': {}}
    model_kwargs_['y']['mask'] = (torch.zeros([1, 1, 1, n_frames]) < 1)     # [..., n_seed:]
    model_kwargs_['y']['audio'] = torch.randn(bs, n_frames - n_seed - n_seed, audio_feature_dim)  # attention5
    # model_kwargs_['y']['audio'] = torch.randn(bs, n_frames - n_seed, audio_feature_dim)       # attention4
    # model_kwargs_['y']['audio'] = torch.randn(bs, n_frames, audio_feature_dim)  # attention3
    model_kwargs_['y']['style'] = torch.randn(bs, style_dim)
    model_kwargs_['y']['mask_local'] = torch.ones(bs, n_frames).bool()
    model_kwargs_['y']['seed'] = x[..., 0:n_seed]       # attention3/4
    model_kwargs_['y']['seed_last'] = x[..., -n_seed:]  # attention5
    model_kwargs_['y']['gesture'] = torch.randn(bs, n_frames, njoints)
    y = model(x, t, model_kwargs_['y'])     # [bs, njoints, nfeats, nframes]
    print(y.shape)