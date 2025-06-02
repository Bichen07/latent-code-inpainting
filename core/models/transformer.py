import os, math
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import random
from train import instantiate_from_config
from core.modules.util import SOSProvider, scatter_mask, box_mask, mixed_mask, RandomMask, BatchRandomMask
import clip # <--- [新增] 導入 clip


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class Transformer(pl.LightningModule):
    def __init__(self,
                 transformer_config,
                 cond_stage_config,
                 first_stage_config=None,
                 refinement_stage_config=None,
                 # ==== [新增] 可以考慮增加參數來控制 CLIP 模型類型 ====
                 clip_model_name="ViT-B/32", # 預設的 CLIP 模型
                 # ===================================================
                 permuter_config=None,
                 ckpt_path=None,
                 ignore_keys=[],
                 first_stage_key="image",
                 cond_stage_key="depth",
                 mask_scheduler='cosine',
                 mask_function="random_mask",
                 mask_start_ratio=0.15,
                 mask_end_ratio=0.75,
                 mask_end_step=600000,
                 mask_on_latent=False,
                 downsample_cond_size=-1,
                 pkeep=0.5,
                 sos_token=0,
                 unconditional=False,
                 attention_extent='mask',
                 ):
        super().__init__()
        self.be_unconditional = unconditional
        self.sos_token = sos_token
        ##########################################
        self.attention_extent = attention_extent
        self.mask_token = -1 # this needs to be hard-coded due to embedding implementation in BERT
        self.mask_start_ratio = mask_start_ratio
        self.mask_end_ratio = mask_end_ratio
        self.mask_end_step = mask_end_step
        self.mask_scheduler = mask_scheduler
        self.mask_on_latent = mask_on_latent
        ##########################################
        self.first_stage_key = first_stage_key
        self.cond_stage_key = cond_stage_key
        self.first_stage_model = None
        self.second_stage_model = None
        self.pkeep = pkeep

        if permuter_config is None:
            permuter_config = {"target": "core.modules.transformer.permuter.Identity"}            
        self.permuter = instantiate_from_config(config=permuter_config)

        # ==== [新增] 初始化 CLIP Text Encoder ====
        self.clip_model = None
        self.text_embedding_dim = None # 先設為 None
        try:
            print(f"Loading CLIP model: {clip_model_name}...")
            # 我們需要將模型移動到 self.device，但此時 self.device 可能還未定義
            # PyTorch Lightning 會自動處理模型到設備的移動，所以先載入到 CPU
            self.clip_model, _ = clip.load(clip_model_name, device="cpu") 
            self.clip_model.requires_grad_(False) # 凍結 CLIP 參數
            self.clip_model.eval()
            # 獲取 CLIP text embedding 的維度
            # 這取決於具體的 CLIP 模型，對於 ViT-B/32 是 512
            # 可以通過 model.text_projection (如果存在) 或 model.ln_final (視覺部分) 的權重形狀推斷
            # 或者直接使用已知值。一個通用的獲取方式是執行一次 encode_text
            dummy_text = clip.tokenize(["test"]).to("cpu")
            with torch.no_grad():
                dummy_embedding = self.clip_model.encode_text(dummy_text)
            self.text_embedding_dim = dummy_embedding.shape[-1]
            print(f"CLIP model '{clip_model_name}' loaded. Text embedding dimension: {self.text_embedding_dim}")
        except Exception as e:
            print(f"Could not load CLIP model '{clip_model_name}': {e}. Text conditioning will be effectively disabled.")
        # =====================================

        # ==== [修改] 實例化 mingpt.GPT 時傳入 context_dim ====
        # 我們需要將 text_embedding_dim 作為 context_dim 傳給 mingpt.GPT
        # 這需要在 transformer_config 的 params 中加入 context_dim
        # 如果 YAML 中已經配置了 context_dim，這裡會使用 YAML 中的值
        # 如果 YAML 中沒有，我們會嘗試使用 self.text_embedding_dim
        
        # 確保 transformer_config.params 存在
        if "params" not in transformer_config:
            transformer_config["params"] = {}
        
        # 如果 YAML 中沒有定義 context_dim，並且我們成功加載了 CLIP，則使用 CLIP 的維度
        if self.clip_model is not None and "context_dim" not in transformer_config["params"]:
            transformer_config["params"]["context_dim"] = self.text_embedding_dim
            print(f"Setting mingpt.GPT context_dim to {self.text_embedding_dim} from CLIP model.")
        elif "context_dim" in transformer_config["params"]:
            print(f"Using context_dim from YAML: {transformer_config['params']['context_dim']}")
            if self.clip_model is not None and self.text_embedding_dim != transformer_config['params']['context_dim']:
                print(f"Warning: CLIP text_embedding_dim ({self.text_embedding_dim}) " +
                      f"does not match context_dim in YAML ({transformer_config['params']['context_dim']}). " +
                      "Ensure this is intended.")
        else: # clip_model is None and context_dim not in YAML
             print("Warning: CLIP model not loaded and context_dim not specified in YAML. Cross-Attention in GPT might not work as expected.")
             # 即使如此，我們仍然可以給 GPT 傳遞一個 context_dim=None，讓它退化到沒有 Cross-Attention
             transformer_config["params"]["context_dim"] = None
        # =====================================

        self.transformer = instantiate_from_config(config=transformer_config)
        self.use_condGPT = self.transformer.__class__.__name__ == 'CondGPT'

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.init_cond_stage_from_ckpt(cond_stage_config)

        if first_stage_config is not None:
            self.init_first_stage_from_ckpt(first_stage_config)

        if refinement_stage_config is not None:
            self.init_full_stages_from_ckpt(refinement_stage_config)

        self.downsample_cond_size = downsample_cond_size
        self.pkeep = pkeep

        # todo: remove hard-coded mapping
        # self.mask_function = self.scatter_mask
        if mask_function == "random_mask":
            # BatchRandomMask(batch_size, s, hole_range=[min_hole_ratio, max_hole_ratio])
            # p_from_scheduler 就是洞的比例 (hole_ratio)
            self.mask_function_impl = lambda z_indices, p_hole_ratio: torch.from_numpy(
                BatchRandomMask(
                    batch_size=z_indices.shape[0],
                    s=int(math.sqrt(z_indices.shape[-1])),
                    hole_range=[p_hole_ratio, p_hole_ratio] 
                )
            ).to(z_indices.device).long()
            self._mask_function_name_for_log = "BatchRandomMask (from core.modules.util)"

        elif mask_function == "box_mask":
            # self.box_mask(shape, device, p_keep_ratio)
            # p_hole_ratio 是遮蔽比例，所以 p_keep_ratio = 1.0 - p_hole_ratio
            self.mask_function_impl = lambda z_indices, p_hole_ratio: self.box_mask(
                                            shape=z_indices.shape,
                                            device=z_indices.device,
                                            p=(1.0 - p_hole_ratio) # p 參數是保留比例
                                        )
            self._mask_function_name_for_log = "box_mask (Transformer class method)"

        elif mask_function == "scatter_mask":
            # self.scatter_mask(shape, device, p_keep_ratio)
            self.mask_function_impl = lambda z_indices, p_hole_ratio: self.scatter_mask(
                                            shape=z_indices.shape,
                                            device=z_indices.device,
                                            p=(1.0 - p_hole_ratio) # p 參數是保留比例
                                        )
            self._mask_function_name_for_log = "scatter_mask (Transformer class method)"
        else:
            print(f"Warning: Unknown mask_function '{mask_function}'. Defaulting to scatter_mask.")
            self.mask_function_impl = lambda z_indices, p_hole_ratio: self.scatter_mask(
                                            shape=z_indices.shape,
                                            device=z_indices.device,
                                            p=(1.0 - p_hole_ratio)
                                        )
            self._mask_function_name_for_log = "scatter_mask (defaulted)"
        
        print(f"Using mask function implementation: {self._mask_function_name_for_log}")


    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        for k in sd.keys():
            for ik in ignore_keys:
                if k.startswith(ik):
                    self.print("Deleting key {} from scos functiontate_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def init_first_stage_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.first_stage_model = model
        self.second_stage_model = None

    def init_full_stages_from_ckpt(self, config):
        model = instantiate_from_config(config)
        model = model.eval()
        model.train = disabled_train
        self.second_stage_model = model
        self.first_stage_model = model.first_stage_model

    def set_first_stage_model(self, model):
        self.first_stage_model = model

    def set_second_stage_model(self, model):
        self.second_stage_model = model

    def init_cond_stage_from_ckpt(self, config):
        if config == "__is_first_stage__":
            print("Using first stage also as cond stage.")
            self.cond_stage_model = self.first_stage_model
        elif config == "__is_unconditional__" or self.be_unconditional:
            print(f"Using no cond stage. Assuming the training is intended to be unconditional. "
                  f"Prepending {self.sos_token} as a sos token.")
            self.be_unconditional = True
            self.cond_stage_key = self.first_stage_key
            self.cond_stage_model = SOSProvider(self.sos_token)
        else:
            model = instantiate_from_config(config)
            model = model.eval()
            model.train = disabled_train
            self.cond_stage_model = model

    def mask_ratio_scheduler(self, B, fix_ratio=None):
        # lienar scheduling function
        # ratio = min(self.mask_start_ratio + B * self.global_step / self.mask_end_step, self.mask_end_ratio)
        if fix_ratio is None:
            ratio = random.uniform(self.mask_start_ratio, self.mask_end_ratio)
            return ratio
        else:
            return fix_ratio

    def mixed_mask(self, z_indices, p=None, scatter_prob=0.5):
        if np.random.rand() < scatter_prob:
            return self.scatter_mask(z_indices, p)
        else:
            return self.box_mask(z_indices, p)

    def scatter_mask(self, z_indices, p=None):
        p = self.pkeep if p is None else (1 - p)
        assert p <= 1 and p >= 0
        mask = torch.bernoulli(p *torch.ones(z_indices.shape,
                                                     device=z_indices.device))
        mask = mask.round().to(dtype=torch.int64) 
        return mask

    def box_mask(self, z_indices, p=None, det=False):
        p = self.pkeep if p is None else p
        assert p <= 1 and p >= 0
        nb = z_indices.shape[0]
        r = int(z_indices.shape[-1]**0.5)
        mr = int(p * r)
        mask = np.ones([nb, r, r]).astype(np.int32)
        for i in range(nb):
            if det:
                h = (r-mr) // 2
                w = h
            else:
                h, w = np.random.randint(0, r-mr, 2)
            mask[i, h:h+mr, w:w+mr] = 0
        mask = torch.from_numpy(mask).reshape(nb, -1).to(z_indices.device)
        return mask

    def preprocess_mask(self, mask, z_indices):
        B = z_indices.shape[0]
        r = int(z_indices.shape[-1]**0.5)
        if len(mask.shape) == 4:
            H = mask.shape[-1]
            if r != H:
                mask = torch.nn.functional.interpolate(mask.float(), scale_factor=r/H)
            mask = mask.reshape(B, -1)
        elif len(mask.shape) == 3:
            assert mask.shape[1] == 1 and mask.shape[-1] == z_indices.shape[-1]
            mask = mask.reshape(B, -1)
        else:
            assert len(mask.shape) == 2 and mask.shape[-1] == z_indices.shape[-1]
        mask = mask.to(dtype=torch.int64)
        return mask

    def forward(self, x, c, mask=None, text_prompts=None): # <--- [修改] 增加 text_prompts 參數
        # one step to produce the logits
        _, z_indices = self.encode_to_z(x.float())
        _, c_indices = self.encode_to_c(c.float())

        # during training, we'll mask out some tokens (0.15 by default)
        assert mask is not None or self.pkeep < 1.0 # 原始斷言
        current_mask = mask # 使用一個新變數名以避免覆蓋參數 mask
        if current_mask is None:
            current_hole_ratio = self.mask_ratio_scheduler(x.shape[0]) 
            current_mask = self.mask_function_impl(z_indices, p_hole_ratio=current_hole_ratio)
        else:
            current_mask = self.preprocess_mask(current_mask, z_indices)

        # !!! replacing masked indices with the [MASK] token (a.k.a -1)
        r_indices = torch.full_like(z_indices, self.mask_token)
        a_indices = current_mask * z_indices + (1 - current_mask) * r_indices           
        a_indices = a_indices + 1 

        # from this point we are making predictions on tokens where MASK==0 
        cz_indices = torch.cat((c_indices, a_indices), dim=1)
        mask_c_indices = torch.full_like(c_indices, 1, dtype=torch.int64).to(mask.device)
        mask_for_gpt_self_attn  = torch.cat([mask_c_indices, mask], dim=1)

        # ==== [新增] 準備文字 context ====
        text_context_features = None
        if self.clip_model is not None and text_prompts is not None:
            # 確保 clip_model 在正確的 device 上 (PyTorch Lightning 會處理 self.clip_model.to(self.device))
            # print(f"Device of clip_model: {next(self.clip_model.parameters()).device}, text_prompts device (example): {x.device}")
            with torch.no_grad():
                # text_prompts 應該是一個 list of strings
                text_tokens = clip.tokenize(text_prompts).to(self.device) # 確保 tokenize 在正確的 device
                text_features = self.clip_model.encode_text(text_tokens).float() 
            # CrossAttention 通常期望 context 是 (B, ContextSeqLen, ContextEmbedDim)
            # 對於單個句子，ContextSeqLen 可以是 1
            text_context_features = text_features.unsqueeze(1) # (B, 1, clip_dim)
        # =================================

        # ==== [修改] self.transformer 的調用，傳入 context ====
        # mingpt.GPT 的 forward 方法簽名應為 forward(self, idx, context=None, ...)
        if self.attention_extent == 'mask':
            logits, _ = self.transformer(idx=cz_indices, mask=mask_for_gpt_self_attn, 
                                         autoregressive=False, context=text_context_features)
        else:
            logits, _ = self.transformer(idx=cz_indices, mask=None, 
                                         autoregressive=False, context=text_context_features)
        # ===================================================

        target = z_indices
        logits = logits[:, c_indices.shape[1]:, 1:] # 原始邏輯：去除條件部分的 logits，並調整詞彙表
        return logits, target, current_mask # 返回 current_mask

    def top_k_logits(self, logits, k):
        v, ix = torch.topk(logits, k)
        out = logits.clone()
        out[out < v[..., [-1]]] = -float('Inf')
        return out

    @torch.no_grad()
    def sample(self, x, c, text_prompts_for_sampling=None, # <--- [新增] text_prompts_for_sampling 參數
               sampling_ratio=0.2, 
               temperature=1.0, 
               sample=True, 
               temperature_degradation=0.9, 
               top_k=None, 
               callback=lambda k: None, 
               scheduler = 'cosine',
               t_scheduler=lambda t,k,d: t*(d**k),
               return_probs=False
               ):

        # scheduler = self.mask_scheduler

        # indices padding
        x = x + 1
        mask = (x == 0) # true: needs update, false: already updated

        if not self.use_condGPT:
            x = torch.cat((c,x),dim=1)

        # ==== [新增] 準備文字 context for sampling ====
        text_context_features_sampling = None
        if self.clip_model is not None and text_prompts_for_sampling is not None:
            with torch.no_grad():
                text_tokens = clip.tokenize(text_prompts_for_sampling).to(self.device)
                text_features = self.clip_model.encode_text(text_tokens).float()
            text_context_features_sampling = text_features.unsqueeze(1)
        # =============================================

        block_size = self.transformer.get_block_size()
        assert not self.transformer.training
        k_unknown = mask[0].sum()
        length = x.size(1)
        steps = int(1 / sampling_ratio) # max(1, k_unknown // n_sample) + 2
        counter = 0

        if scheduler == 'cosine':
            unknowns = (k_unknown.item() * np.cos(math.pi * np.arange(steps) / (2 * steps))).astype(np.int32)
            drawn = unknowns - np.append(unknowns, 0)[1:]

        prob_results = []
        gathered_idx_info = []
        
        while True:
            assert x.size(1) <= block_size # make sure model can see conditioning

            # break if we have no candidate left
            if mask.sum() == 0:
                # print(f"sampling complete at step {k}/{steps}")
                break

            # scheduling function to determine how much to sample at each step
            if scheduler == 'cosine':
                # consine sampling
                n_sample = max(drawn[counter] if counter < len(drawn) else drawn[-1], 1)
            else:
                # default with linear scheduling
                n_sample = int(k_unknown.item() * sampling_ratio)

            t = t_scheduler(temperature, counter, temperature_degradation)

            counter += 1            
            mask_c_indices = torch.full_like(c, 0, dtype=torch.int64).to(x.device)
            mask_c = torch.cat([mask_c_indices, mask], dim=1)
            x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed

            # if self.use_condGPT:
            #     logits, _ = self.transformer(x_cond, c, mask=(1-mask_c.int()), autoregressive=False)

            # if self.attention_extent == 'mask':
            #     logits, _ = self.transformer(x_cond, mask=(1-mask_c.int()), autoregressive=False)
            # else:
            #     logits, _ = self.transformer(x_cond, mask=None, autoregressive=False)

            if self.attention_extent == 'mask':
                logits, _ = self.transformer(idx=x_cond, mask=(1-mask_c.int()),  # <--- [修改] 傳入 context
                                             autoregressive=False, context=text_context_features_sampling)
            else:
                logits, _ = self.transformer(idx=x_cond, mask=None,             # <--- [修改] 傳入 context
                                             autoregressive=False, context=text_context_features_sampling)
            # pluck the logits at the final step and scale by temperature              
            logits = logits[:, c.shape[1]:, 1:] / t

            # optionally crop probabilities to only the top k options
            if top_k is not None:
                logits = self.top_k_logits(logits, top_k)

            # apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)

            # prune low confidence prediction (we are taking the top% confidence score ones and remove the rest)
            B, L, D = probs.shape
            k = min(n_sample, 1 + mask.sum()//B) # min(max(L // steps, 4), mask.sum()//B + 1)
            callback(k)

            # print(f"taking top {k} from the unpredicted {mask.sum()//B}")
            # gathering k most confident samples
            max_probs, _ = torch.max(probs * mask[...,None], -1)
            # B, k
            k_candidates = torch.argsort(max_probs, 1, descending=True)[:,:k]
            # B, L, D -> B, k, D
            gathered_probs = torch.gather(probs, 1, k_candidates[...,None].expand(-1,-1,D)) 

            if sample:
                gathered_idx = torch.multinomial(gathered_probs.reshape(-1,D),num_samples=1).reshape(B,k)
            else:
                _, gathered_idx = torch.topk(gathered_probs, k=1, dim=-1)
                gathered_idx = gathered_idx[..., -1]

            #### FOR DEBUG ONLY #####
            if return_probs:
                select_probs = torch.gather(gathered_probs, 2, gathered_idx.unsqueeze(-1))
                prob_results.append(select_probs.detach().cpu().numpy())
                k_candidates_info = (k_candidates).detach().cpu().numpy()
                gathered_idx_info.append(k_candidates_info)
            #############################

            # updating mask and indices
            for i in range(B):
                mask[i, k_candidates[i]] = False
                x[i, (k_candidates[i]+1)] = gathered_idx[i] + 1


        # cut off conditioning
        if not self.use_condGPT:
            x = x[:, c.shape[1]:]

        if return_probs:
            return x - 1, prob_results, gathered_idx_info

        return x - 1

    @torch.no_grad()
    def encode_to_z(self, x, mask=None):
        if mask is not None:
            quant_z, _, info, mask_out = self.first_stage_model.encode(x, mask)
        else:
            quant_z, _, info = self.first_stage_model.encode(x)

        indices = info[2].view(quant_z.shape[0], -1)
        indices = self.permuter(indices)

        if mask is None:
            return quant_z, indices
        else:
            return quant_z, indices, mask_out

    @torch.no_grad()
    def encode_to_c(self, c):
        if self.downsample_cond_size > -1:
            c = F.interpolate(c, size=(self.downsample_cond_size, self.downsample_cond_size))
        quant_c, _, [_,_,indices] = self.cond_stage_model.encode(c)
        if self.use_condGPT or len(indices.shape) > 2:
            indices = indices.view(c.shape[0], -1)           
        return quant_c, indices

    @torch.no_grad()
    def decode_to_img(self, index, zshape, use_softmax=False, return_quant=False):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)

        x = self.first_stage_model.decode(quant_z)

        if use_softmax or self.first_stage_key != 'image':
            x = F.softmax(x, dim=1)
        if return_quant:
            return x, quant_z
        else:
            return x

    @torch.no_grad()
    def decode_to_quant(self, index, zshape, use_softmax=False):
        index = self.permuter(index, reverse=True)
        bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
        quant_z = self.first_stage_model.quantize.get_codebook_entry(
            index.reshape(-1), shape=bhwc)
        return quant_z

    @torch.no_grad()
    def forward_to_recon(self, batch, mask=None, det=False, return_quant=False):
        '''
        similar to log image, but return a single image tensor given the default mask function instead of logging results

        '''
        # x, c = self.get_xc(batch)

        # ==== [修改] 獲取 text_prompts ====
        xc_output = self.get_xc(batch)
        if len(xc_output) == 3:
            x, c, text_prompts = xc_output
        else: # 向下兼容或沒有 text_prompt 的情況
            x, c = xc_output
            text_prompts = None
        # ================================

        x = x.to(device=self.device).float()
        c = c.to(device=self.device).float()
        quant_z, z_indices = self.encode_to_z(x)
        quant_c, c_indices = self.encode_to_c(c)
        B, C, H, W = x.shape
        gH, gW = quant_z.shape[2:]

        if mask is None:
            mask = self.mask_function(z_indices)
        else:
            mask = self.preprocess_mask(mask, z_indices)

        r_indices = torch.full_like(z_indices, self.mask_token)
        z_start_indices = mask*z_indices+(1-mask)*r_indices      
        # index_sample = self.sample(z_start_indices, 
        #                            c_indices,
        #                            sample= not det)
        # ==== [修改] 調用 sample 時傳入 text_prompts ====
        index_sample = self.sample(z_start_indices,  #
                                   c_indices, #
                                   text_prompts_for_sampling=text_prompts, # <--- 新增
                                   sample= not det) #
        # ===========================================
        return self.decode_to_img(index_sample, quant_z.shape, return_quant=return_quant)

    @torch.no_grad()
    def forward_to_indices(self, batch, z_indices, mask, det=False):
        # x, c = self.get_xc(batch)
        # ==== [修改] 獲取 text_prompts ====
        xc_output = self.get_xc(batch) #
        if len(xc_output) == 3: #
            x, c, text_prompts = xc_output #
        else: #
            x, c = xc_output #
            text_prompts = None #
        # ================================
        x = x.to(device=self.device).float()
        c = c.to(device=self.device).float()
       
        # quant_z, z_indices = self.encode_to_z(x)
        # B, C, H, W = x.shape
        quant_c, c_indices = self.encode_to_c(c)
        
        mask = self.preprocess_mask(mask, z_indices)
        r_indices = torch.full_like(z_indices, self.mask_token)
        z_start_indices = mask*z_indices+(1-mask)*r_indices      
        # index_sample = self.sample(z_start_indices.to(device=self.device), 
        #                            c_indices.to(device=self.device),
        #                            sample= not det)
        # ==== [修改] 調用 sample 時傳入 text_prompts ====
        index_sample = self.sample(z_start_indices.to(device=self.device),  #
                                   c_indices.to(device=self.device), #
                                   text_prompts_for_sampling=text_prompts, # <--- 新增
                                   sample= not det) #
        # ===========================================
        return index_sample


    @torch.no_grad()
    def log_images(self, batch, temperature=None, top_k=None, callback=None, lr_interface=False, composition=True, **kwargs):
        
        log = dict()
        # x, c = self.get_xc(batch)
        # ==== [修改] 獲取 text_prompts ====
        xc_output = self.get_xc(batch) #
        if len(xc_output) == 3: #
            x, c, text_prompts = xc_output #
        else: #
            x, c = xc_output #
            text_prompts = None #
        # ================================
        x = x.to(device=self.device).float()
        c = c.to(device=self.device).float()

        B, C, H, W = x.shape
        pkeep = self.mask_ratio_scheduler(B, fix_ratio=0.5)       

        if not self.mask_on_latent:
            # mask_in = 1 - box_mask(x.shape, x.device, pkeep, det=True)
            mask_in = torch.from_numpy(BatchRandomMask(x.shape[0], x.shape[-1])).to(x.device)
            quant_z, z_indices, mask_out = self.encode_to_z(x * mask_in, mask_in)
        else:
            quant_z, z_indices = self.encode_to_z(x)

        quant_c, c_indices = self.encode_to_c(c)
        gH, gW = quant_z.shape[2:]

        # inpainting sample
        if self.mask_on_latent:
            mask = self.box_mask(z_indices, p=pkeep, det=True)
            image_mask = torch.nn.functional.interpolate(mask.reshape(B,1,gH,gW).float(), scale_factor=H//gH)
            z_indices_recon = z_indices
        else:
            image_mask = mask_in
            mask = mask_out.reshape(mask_out.shape[0], -1).int()
            _, z_indices_recon = self.encode_to_z(x)
       
        r_indices = torch.full_like(z_indices, self.mask_token)
        z_start_indices = mask*z_indices+(1-mask)*r_indices      

        # # using default sampling setting
        # index_sample = self.sample(z_start_indices, 
        #                            c_indices)
        # ==== [修改] 調用 sample 時傳入 text_prompts ====
        index_sample = self.sample(z_start_indices,  #
                                   c_indices, #
                                   text_prompts_for_sampling=text_prompts) # <--- 新增
        # ===========================================
        ####################
        # index_sample = z_indices * mask + (1-mask) * z_indices_gt
        #####################

        x_sample_det = self.decode_to_img(index_sample, quant_z.shape)

        # composition
        if composition:
            x_sample_det_comp = image_mask * x + (1 - image_mask) * x_sample_det
            if self.second_stage_model is not None:
                x_sample_det_comp = self.second_stage_model.refine(x_sample_det_comp, image_mask)

        x_masked = image_mask * x

        # reconstruction
        # if self.mask_on_latent:
        #     x_rec = self.decode_to_img(z_indices_recon, quant_z.shape)
        # else:
        #     r_indices = torch.full_like(z_indices_recon, self.mask_token)
        #     z_start_indices = mask*z_indices_recon+(1-mask)*r_indices      
        #     index_sample = self.sample(z_start_indices, c_indices,
        #                                sample=False,
        #                                callback=callback if callback is not None else lambda k: None)
        #     x_rec = self.decode_to_img(index_sample, quant_z.shape)
        if self.mask_on_latent: #
            x_rec = self.decode_to_img(z_indices_recon, quant_z.shape) #
        else: #
            r_indices_rec = torch.full_like(z_indices_recon, self.mask_token) #
            z_start_indices_rec = mask*z_indices_recon+(1-mask)*r_indices_rec #     
            index_sample_rec = self.sample(z_start_indices_rec, c_indices, # <--- [修改] 此處也可能需要 text_prompts
                                           text_prompts_for_sampling=text_prompts, #
                                           sample=False, #
                                           callback=callback if callback is not None else lambda k: None) #
            x_rec = self.decode_to_img(index_sample_rec, quant_z.shape) #

        log["samples_det_gen"] = x_sample_det #
        if composition: #
            x_sample_det_comp = image_mask * x + (1 - image_mask) * x_sample_det #
            if hasattr(self, 'second_stage_model') and self.second_stage_model is not None: # 確保 second_stage_model 存在
                x_sample_det_comp = self.second_stage_model.refine(x_sample_det_comp, image_mask) #
            log["samples_det"] = x_sample_det_comp #
        log["inputs_masked"] = image_mask * x # 使用 image_mask 而不是 x_masked (如果 x_masked 未定義)

        # log["inputs"] = x
        # log["reconstructions"] = x_rec

        if self.be_unconditional:
            pass
        elif self.cond_stage_key != "image":
            cond_rec = self.cond_stage_model.decode(quant_c)
            if self.cond_stage_key == "segmentation":
                # get image from segmentation mask
                num_classes = cond_rec.shape[1]

                c = torch.argmax(c, dim=1, keepdim=True)
                c = F.one_hot(c, num_classes=num_classes)
                c = c.squeeze(1).permute(0, 3, 1, 2).float()
                c = self.cond_stage_model.to_rgb(c)

                cond_rec = torch.argmax(cond_rec, dim=1, keepdim=True)
                cond_rec = F.one_hot(cond_rec, num_classes=num_classes)
                cond_rec = cond_rec.squeeze(1).permute(0, 3, 1, 2).float()
                cond_rec = self.cond_stage_model.to_rgb(cond_rec)
            log["conditioning_rec"] = cond_rec
            log["conditioning"] = c

        # log["samples_half"] = x_sample_half
        # log["sample_scatter"] = x_sample_scatter_comp
        # log["sample_scatter_gen"] = x_sample_scatter
        log["samples_det_gen"] = x_sample_det
        log["samples_det"] = x_sample_det_comp
        log["inputs_masked"] = x_masked

        return log

    def get_input(self, key, batch):
        x = batch[key]
        if len(x.shape) == 3:
            x = x[..., None]
        if len(x.shape) == 4:
            x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        if x.dtype == torch.double:
            x = x.float()
        return x

    # def get_xc(self, batch, N=None):
    #     x = self.get_input(self.first_stage_key, batch)
    #     c = self.get_input(self.cond_stage_key, batch)
    #     if N is not None:
    #         x = x[:N]
    #         c = c[:N]
    #     return x, c

    def get_xc(self, batch, N=None): #
        x = self.get_input(self.first_stage_key, batch) #
        c = self.get_input(self.cond_stage_key, batch) #
        
        # ==== [新增] 獲取 text_prompts ====
        text_prompts = batch.get("text_prompt", None) # 從 batch 中獲取，如果沒有則為 None
        # text_prompts 應該是一個字串列表，長度為 batch_size
        # ===============================
            
        if N is not None: #
            x = x[:N] #
            c = c[:N] #
            if text_prompts is not None: #
                text_prompts = text_prompts[:N] #
        
        if text_prompts is not None: #
            return x, c, text_prompts # 返回三元組 #
        else: # 如果沒有 text_prompts，則返回原始的二元組，或者 x,c,None
            return x, c, None #

    # loss function defined here
    def shared_step(self, batch, batch_idx):
        # x, c = self.get_xc(batch)
        # ==== [修改] 獲取 text_prompts ====
        xc_output = self.get_xc(batch) #
        if len(xc_output) == 3: #
            x, c, text_prompts = xc_output #
        else: #
            x, c = xc_output #
            text_prompts = None #
        # ================================
        # logits, target, mask = self(x, c) # 原有
        # ==== [修改] 傳遞 text_prompts 給 forward ====
        logits, target, mask = self(x, c, text_prompts=text_prompts) # <--- 新增傳遞
        # ============================================

        B, L, N = logits.shape
        mask_reverse = (1 - mask).bool()
        target_select = torch.masked_select(target.reshape(-1), mask_reverse.reshape(-1))
        logits_select = torch.masked_select(logits.reshape(-1, N), mask_reverse.reshape(-1,1))
        logits_select = logits_select.reshape(-1, N)
        # loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
        loss = F.cross_entropy(logits_select, target_select)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def configure_optimizers_with_lr(self, lr):
        """
        Following minGPT:
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.transformer.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.transformer.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.01},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95))
        return optimizer

    def configure_optimizers(self):
        return self.configure_optimizers_with_lr(self.learning_rate)