from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from .. import LLMBaseEditor, BaseLLMForEdit
from dataclasses import dataclass
from ... import BaseConfig
from typing import List, Literal
from .utils import nethook
from .utils.generate import generate_fast
from datasets import load_dataset
from tqdm import tqdm
import os

COV_CACHE = {}
STATS_DIR = 'data/rome-memit-stats'


@dataclass
class AlphaEditConfig(BaseConfig):
    edit_model_name: str
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"]
    v_num_grad_steps: int
    v_lr: float
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    kl_factor: float
    mom2_adjustment: bool
    mom2_update_weight: float

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    nullspace_threshold: float
    L2: float


class AlphaEdit(LLMBaseEditor):
    def __init__(self, llm: BaseLLMForEdit, config:AlphaEditConfig, device = 'cuda', 
                 verbose = False, wiki_path = 'data/wiki-for-rome-memit/wikipedia-20200501.en'):
        super().__init__(llm, device)
        self.cfg = config
        self.verbose = verbose
        self.wiki_path = wiki_path
        # Get alpha_edit cache
        W_out = nethook.get_parameter(self.llm.model, f"{config.rewrite_module_tmp.format(config.layers[-1])}.weight")
        if config.edit_model_name == "gpt2-xl":
            self.cache_c = torch.zeros((len(config.layers), W_out.shape[0], W_out.shape[0]), device = device)
            self.P = torch.zeros((len(config.layers), W_out.shape[0], W_out.shape[0]), device = device)
        elif config.edit_model_name in ["gpt-j-6b", "llama-3-8b", "phi-1.5"]:
            self.cache_c = torch.zeros((len(config.layers), W_out.shape[1], W_out.shape[1]), device = device)
            self.P = torch.zeros((len(config.layers), W_out.shape[1], W_out.shape[1]), device = device)
        else:
            raise
        null_space_project_save_dir = 'data/alphaedit/null_space_project'
        os.makedirs(null_space_project_save_dir, exist_ok=True)
        null_space_project_save_path = os.path.join(null_space_project_save_dir, config.edit_model_name)
        if os.path.exists(null_space_project_save_path):
            self.P = torch.load(null_space_project_save_path, map_location=torch.device(self.device))
        else:
            for i, layer in enumerate(config.layers):
                self.P[i,:,:] = self.get_project(layer)
            torch.save(self.P, null_space_project_save_path)
        # Cache original weights
        self.original_weights = {
            f"{self.cfg.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
                self.llm.model, f"{self.cfg.rewrite_module_tmp.format(layer)}.weight"
            ).detach().clone() for layer in self.cfg.layers
        }
        # subject placeholder
        self.subject_placeholder = 'adf-+g^&a+sda-ad4*+54a@'
        self.get_context_templates()

    def name_of_editor_and_model(self)->Tuple[str, str]:
        return 'alphaedit', self.cfg.edit_model_name

    def if_can_batch_edit(self):
        return False

    def edit_one_piece(self, request: Dict) -> None:
        """request = {'prompt':str, 'subject':str, 'target_new':str}"""
        self.edit([request])

    def edit_batch(self):
        raise 

    def restore_to_original_model(self):
        with torch.no_grad():
            self.cache_c *= 0
            for layer in self.cfg.layers:
                w_name = f"{self.cfg.rewrite_module_tmp.format(layer)}.weight"
                w = nethook.get_parameter(self.llm.model, w_name)
                w *= 0
                w += self.original_weights[w_name]

    def save_current_edit_status(self):
        self.saved_edit_status_cache_c = self.cache_c.detach().clone()
        self.saved_edit_status_weights = {
            f"{self.cfg.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
                self.llm.model, f"{self.cfg.rewrite_module_tmp.format(layer)}.weight"
            ).detach().clone() for layer in self.cfg.layers
        }
        
    def restore_to_saved_edit_status(self):
        self.cache_c = self.saved_edit_status_cache_c.detach().clone()
        with torch.no_grad():
            for layer in self.cfg.layers:
                w_name = f"{self.cfg.rewrite_module_tmp.format(layer)}.weight"
                w = nethook.get_parameter(self.llm.model, w_name)
                w *= 0
                w += self.saved_edit_status_weights[w_name]


    ###########################################################################
    ###################### AlphaEdit Functions  ###############################
    ###########################################################################
    def edit(self, requests: List[Dict]) -> Dict[str, Tuple[torch.Tensor]]:
        """
        Executes the MEMIT update algorithm for the specified update at the specified layer
        Invariant: model at beginning of function == model at end of function
        """
        # Update target and print info
        requests = deepcopy(requests)
        for i, request in enumerate(requests):
            request['prompt'] = request['prompt'].replace(request['subject'], self.subject_placeholder, 1) 
            request["target_new"] = {"str": request["target_new"]}
            if request["target_new"]["str"][0] != " ":
                # Space required for correct tokenization
                requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
        if self.verbose:
            for request in requests[:10]:
                print(
                    f"MEMIT request sample: "
                    f"[{request['prompt'].replace(self.subject_placeholder, request['subject'])}] -> [{request['target_new']['str']}]"
                )

        # Retrieve weights that user desires to change
        weights = {
            f"{self.cfg.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
                self.llm.model, f"{self.cfg.rewrite_module_tmp.format(layer)}.weight"
            )
            for layer in self.cfg.layers
        }
        # Compute z for final layer
        z_layer = self.cfg.layers[-1]
        z_list = []

        for request in requests:
            # Retrieve k/v pair if already stored in cache
            data_loaded = False
            # Compute k/v pair if not loaded from cache
            if not data_loaded:
                cur_z = self.compute_z(request, z_layer)

                z_list.append(cur_z)

        zs = torch.stack(z_list, dim=1)

        for i, layer in enumerate(self.cfg.layers):
            if self.verbose:
                print(f"\n\nLAYER {layer}\n")

            # Get current model activations
            layer_ks = self.compute_ks(requests, layer).T
            if self.verbose:
                print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

            # Compute residual error
            cur_zs = self.get_module_input_output_at_words(z_layer,
                [request["prompt"] for request in requests],
                [request["subject"] for request in requests],
                self.cfg.layer_module_tmp, self.cfg.fact_token)[1].T
            targets = zs - cur_zs
            if self.verbose:
                print("z error", torch.linalg.norm(targets, dim=0).mean())

            repeat_factor = (layer_ks.size(1) // targets.size(1))
            targets = targets.repeat_interleave(repeat_factor, dim=1)
            resid = targets / (len(self.cfg.layers) - i)  # Distribute residual across layers
            upd_matrix = torch.linalg.solve(
                self.P[i,:,:].to(self.device) @ (layer_ks @ layer_ks.T + self.cache_c[i,:,:].to(self.device)) + 
                self.cfg.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device=self.device), 
                self.P[i,:,:].to(self.device) @ layer_ks @ resid.T
            )
            # Adjust update matrix shape
            weight_name = f"{self.cfg.rewrite_module_tmp.format(layer)}.weight"
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
            if self.verbose:
                print("orig norm", torch.linalg.norm(weights[weight_name]))
                print("upd norm", torch.linalg.norm(upd_matrix))
            with torch.no_grad():
                weights[weight_name][...] = weights[weight_name] + upd_matrix
            # Clear GPU memory
            #del U,S,cov
            for x in [layer_ks, cur_zs, targets, upd_matrix]:
                x.cpu()
                del x
            torch.cuda.empty_cache()
        for i, layer in enumerate(self.cfg.layers):
            layer_ks = self.compute_ks(requests, layer).T
            self.cache_c[i,:,:] += layer_ks.to(self.device) @ layer_ks.to(self.device).T

        if self.verbose:
            print(f"Deltas successfully computed for {list(weights.keys())}")

    def get_context_templates(self):
        self.CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_fast(
                    self.llm.model,
                    self.llm.tokenizer,
                    ["The", "Therefore", "Because", "I", "You"],
                    n_gen_per_prompt=n_gen // 5,
                    max_out_len=length,
                )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
        ]
        print(f"Cached context templates {self.CONTEXT_TEMPLATES_CACHE}")

    def compute_ks(self,requests: Dict, layer: int):
        hparams = self.cfg
        layer_ks = self.get_module_input_output_at_words(
            layer,
            [
                context.format(request["prompt"])
                for request in requests
                for context_type in self.CONTEXT_TEMPLATES_CACHE
                for context in context_type
            ],
            [
                request["subject"]
                for request in requests
                for context_type in self.CONTEXT_TEMPLATES_CACHE
                for _ in context_type
            ],
            hparams.rewrite_module_tmp, hparams.fact_token,
        )[0]

        context_type_lens = [0] + [len(context_type) for context_type in self.CONTEXT_TEMPLATES_CACHE]
        context_len = sum(context_type_lens)
        context_type_csum = np.cumsum(context_type_lens).tolist()

        ans = []
        for i in range(0, layer_ks.size(0), context_len):
            tmp = []
            for j in range(len(context_type_csum) - 1):
                start, end = context_type_csum[j], context_type_csum[j + 1]
                tmp.append(layer_ks[i + start : i + end].mean(0))
            ans.append(torch.stack(tmp, 0).mean(0))
        return torch.stack(ans, dim=0)


    def compute_z(self, request: Dict, layer: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Computes the value (right) vector for the rank-1 update.
            Runs a simple optimization procedure."""
        model = self.llm.model
        hparams = self.cfg
        tok = self.llm.tokenizer
        # Get model parameters
        lm_w, ln_f = (
            nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
            nethook.get_module(model, hparams.ln_f_module),
        )
        try:
            lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
        except LookupError as _:
            lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)
        if self.verbose:
            print("Computing right vector (v)")

        # Tokenize target into list of int token IDs
        target_ids = tok(request["target_new"]["str"], return_tensors="pt").to(self.device)[
            "input_ids"
        ][0]

        if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
            target_ids = target_ids[1:]
        # Compile list of rewriting and KL x/y pairs
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in self.CONTEXT_TEMPLATES_CACHE
            for context in context_types
        ], ["%s is a"%self.subject_placeholder]
        all_prompts = rewriting_prompts + kl_prompts

        input_tok = tok(
            [prompt.replace(self.subject_placeholder, request["subject"]) 
            for prompt in all_prompts],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Compute rewriting targets
        rewriting_targets = torch.tensor(-100, device=self.device).repeat(
            len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
        )
        for i in range(len(rewriting_prompts)):
            ex_len = input_tok["attention_mask"][i].sum()
            rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

        # Compute indices of the tokens where the fact is looked up
        lookup_idxs = [self.find_fact_lookup_idx(prompt, request["subject"], 
            hparams.fact_token) for i, prompt in enumerate(all_prompts)]

        # Finalize rewrite and loss layers
        loss_layer = max(hparams.v_loss_layer, layer)
        if self.verbose:
            print(f"Rewrite layer is {layer}")
            print(f"Tying optimization objective to {loss_layer}")

        # Set up an optimization over a latent vector that, when output at the
        # rewrite layer, i.e. hypothesized fact lookup location, will induce the
        # target token to be predicted at the final layer.
        if hasattr(model.config, 'n_embd'):
            delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=self.device)
        elif hasattr(model.config, 'hidden_size'):
            delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=self.device)
        else:
            raise NotImplementedError
        target_init, kl_distr_init = None, None

        # Inserts new "delta" variable at the appropriate part of the computation
        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init

            if cur_layer == hparams.layer_module_tmp.format(layer):
                # Store initial value of the vector of interest
                if target_init is None:
                    if self.verbose:
                        print("Recording initial value of v*")
                    # Initial value is recorded for the clean sentence
                    target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

                # Add intervened delta
                for i, idx in enumerate(lookup_idxs):

                    if len(lookup_idxs)!=len(cur_out[0]):
                        cur_out[0][idx, i, :] += delta
                    else:
                        cur_out[0][i, idx, :] += delta

            return cur_out

        # Optimizer
        opt = torch.optim.Adam([delta], lr=hparams.v_lr)
        nethook.set_requires_grad(False, model)

        # Execute optimization
        for it in range(hparams.v_num_grad_steps):
            opt.zero_grad()

            # Forward propagation
            with nethook.TraceDict(
                module=model,
                layers=[
                    hparams.layer_module_tmp.format(loss_layer),
                    hparams.layer_module_tmp.format(layer),
                ],
                retain_input=False,
                retain_output=True,
                edit_output=edit_output_fn,
            ) as tr:
                logits = model(**input_tok).logits

                # Compute distribution for KL divergence
                kl_logits = torch.stack(
                    [
                        logits[i - len(kl_prompts), idx, :]
                        for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                    ],
                    dim=0,
                )
                kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
                if kl_distr_init is None:
                    kl_distr_init = kl_log_probs.detach().clone()

            # Compute loss on rewriting targets
            output=tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
            if output.shape[1]!=rewriting_targets.shape[1]:
                output=torch.transpose(output, 0, 1)
            full_repr = output[:len(rewriting_prompts)]

            log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
                kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
            )
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
            loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
            if self.verbose:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
                    f"avg prob of [{request['target_new']['str']}] "
                    f"{torch.exp(-nll_loss_each).mean().item()}"
                )
            if loss < 5e-2:
                break

            if it == hparams.v_num_grad_steps - 1:
                break

            # Backpropagate
            loss.backward()
            opt.step()

            # Project within L2 ball
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if delta.norm() > max_norm:
                with torch.no_grad():
                    delta[...] = delta * max_norm / delta.norm()

        target = target_init + delta
        if self.verbose:
            print(
                f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
            )

        return target


    def get_project(self, layer):
        hparams = self.cfg
        force_recompute = False
        cov = self.get_cov(
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        ).cpu()
        U, S, _ = torch.linalg.svd(cov, full_matrices=False)
        threshold = hparams.nullspace_threshold
        small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
        print(len(small_singular_indices))
        return U[:, small_singular_indices] @ U[:, small_singular_indices].T

    def get_cov(self, layer_name: str, mom2_dataset: str, mom2_n_samples: str,
            mom2_dtype: str, inv: bool = False, force_recompute: bool = False) -> torch.Tensor:
        """ Retrieves covariance statistics, then computes the algebraic inverse.
            Caches result for future use. """
        model = self.llm.model

        model_name = model.config._name_or_path.replace("/", "_")
        key = (model_name, layer_name)

        print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
        if key not in COV_CACHE or force_recompute:
            stat = self.layer_stats(layer_name, STATS_DIR, mom2_dataset, to_collect=["mom2"],
                sample_size=mom2_n_samples, precision=mom2_dtype, force_recompute=force_recompute)
            COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

        return torch.inverse(COV_CACHE[key]) if inv else COV_CACHE[key]


    def layer_stats(self, layer_name, stats_dir, ds_name, to_collect, model_name=None,
        sample_size=None, precision=None, batch_tokens=None, download=True, progress=tqdm,
        force_recompute=False, hparams=None):
        """Function to load or compute cached stats. """
        from .utils.runningstats import CombinedStat, Mean, NormMean, SecondMoment, tally
        from .utils.tok_dataset import TokenizedDataset, dict_to_, flatten_masked_batch, length_collation
        STAT_TYPES = {
            "mom2": SecondMoment,
            "mean": Mean,
            "norm_mean": NormMean,
        }
        model = self.llm.model
        tokenizer = self.llm.tokenizer
        def get_ds():
            # Load_From_File
            from datasets import Dataset
            raw_ds = Dataset.from_file(os.path.join(self.wiki_path, 'wikipedia-train.arrow'))
            raw_ds = {'train': raw_ds}
            # raw_ds = load_dataset(
            #     ds_name,
            #     dict(wikitext="wikitext-103-raw-v1", wikipedia="20200501.en")[ds_name]
            # )
            if hasattr(model.config, 'n_positions'):
                maxlen = model.config.n_positions
            elif hasattr(model.config, 'max_sequence_length'):
                maxlen = model.config.max_sequence_length
            elif hasattr(model.config, 'max_position_embeddings'):
                maxlen = model.config.max_position_embeddings
            elif hasattr(model.config,'seq_length'):
                maxlen = model.config.seq_length
            else:
                raise NotImplementedError
                    
            if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
                if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                    maxlen = model.config.sliding_window or 4096
                else:
                    maxlen = 4096
            if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
                maxlen = 4096

            if batch_tokens is not None and batch_tokens < maxlen:
                maxlen = batch_tokens
            return TokenizedDataset(raw_ds["train"], tokenizer, maxlen=maxlen)

        # Continue with computation of statistics
        batch_size = 1  # Examine this many dataset texts at once
        if hasattr(model.config, 'n_positions'):
            npos = model.config.n_positions
        elif hasattr(model.config, 'max_sequence_length'):
            npos = model.config.max_sequence_length
        elif hasattr(model.config, 'max_position_embeddings'):
            npos = model.config.max_position_embeddings
        elif hasattr(model.config,'seq_length'):
            npos = model.config.seq_length
        else:
            raise NotImplementedError
            
        if hasattr(model.config, 'model_type') and 'mistral' in model.config.model_type:
            if hasattr(model.config, 'sliding_window') and model.config.sliding_window:
                npos = model.config.sliding_window or 4096
            else:
                npos = 4096
        if hasattr(model.config, 'model_type') and 'qwen2' in model.config.model_type:
                npos = 4096

        if batch_tokens is None:
            batch_tokens = npos * 3  # Sort and divide into batches with this many tokens
        if precision is None:
            precision = "float64"
        dtype = getattr(torch, precision)
        size_suffix = "" if sample_size is None else f"_{sample_size}"
        if batch_tokens < npos:
            size_suffix = "_t{batch_tokens}" + size_suffix
        if model_name is None:
            # model_name = model.config._name_or_path.replace("/", "_")
            model_name = model.config._name_or_path.rsplit("/")[-1]

        stats_dir = Path(stats_dir)
        file_extension = f"{model_name}/{ds_name}_stats/{layer_name}_{precision}_{'-'.join(sorted(to_collect))}{size_suffix}.npz"
        filename = stats_dir / file_extension

        print(f"Computing Cov locally....")

        ds = get_ds() if not filename.exists() else None
        if progress is None:
            progress = lambda x: x

        stat = CombinedStat(**{k: STAT_TYPES[k]() for k in to_collect})
        loader = tally(
            stat,
            ds,
            cache=(filename if not force_recompute else None),
            sample_size=sample_size,
            batch_size=batch_size,
            collate_fn=length_collation(batch_tokens),
            pin_memory=True,
            random_sample=1,
            num_workers=2,
        )
        batch_count = -(-(sample_size or len(ds)) // batch_size)
        with torch.no_grad():
            for batch_group in progress(loader, total=batch_count):
                for batch in batch_group:
                    batch = dict_to_(batch, self.device)
                    with nethook.Trace(
                        model, layer_name, retain_input=True, retain_output=False, stop=True
                    ) as tr:
                        model(**batch)
                    feats = flatten_masked_batch(tr.input, batch["attention_mask"])
                    # feats = flatten_masked_batch(tr.output, batch["attention_mask"])
                    feats = feats.to(dtype=dtype)
                    stat.add(feats)
        return stat

    def get_words_idxs_in_templates(self, context_templates: str, words: str, subtoken: str) -> int:
        """Given list of template strings, each with *one* format specifier
        (e.g. "{} plays basketball"), and words to be substituted into the
        template, computes the post-tokenization index of their last tokens."""
        tok = self.llm.tokenizer

        assert all(tmp.count(self.subject_placeholder) == 1 for tmp in context_templates
            ), "We currently do not support multiple fill-ins for context"

        prefixes_len, words_len, suffixes_len, inputs_len = [], [], [], []
        for i, context in enumerate(context_templates):
            prefix, suffix = context.split(self.subject_placeholder)
            prefix_len = len(tok.encode(prefix))
            prompt_len = len(tok.encode(prefix + words[i]))
            input_len = len(tok.encode(prefix + words[i] + suffix))
            prefixes_len.append(prefix_len)
            words_len.append(prompt_len - prefix_len)
            suffixes_len.append(input_len - prompt_len)
            inputs_len.append(input_len)

        # Compute indices of last tokens
        if subtoken == "last" or subtoken == "first_after_last":
            return [
                [
                    prefixes_len[i]
                    + words_len[i]
                    - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
                ]
                # If suffix is empty, there is no "first token after the last".
                # So, just return the last token of the word.
                for i in range(len(context_templates))
            ]
        elif subtoken == "first":
            return [[prefixes_len[i] - inputs_len[i]] for i in range(len(context_templates))]
        else:
            raise ValueError(f"Unknown subtoken type: {subtoken}")

    def find_fact_lookup_idx(self, prompt: str, subject: str, fact_token_strategy: str) -> int:
        """
        Computes hypothesized fact lookup index given a sentence and subject.
        """
        tok = self.llm.tokenizer

        ret = None
        if fact_token_strategy == "last":
            ret = -1
        elif (
            "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
        ):
            ret = self.get_words_idxs_in_templates([prompt], [subject],
                fact_token_strategy[len("subject_") :])[0][0]
        else:
            raise ValueError(f"fact_token={fact_token_strategy} not recognized")

        sentence = prompt.replace(self.subject_placeholder, subject)
        if self.verbose:
            print(
                f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
                tok.decode(tok(sentence)["input_ids"][ret]),
            )

        return ret

    def get_module_input_output_at_words(self, layer: int, context_templates: List[str],
        words: List[str], module_template: str, fact_token_strategy: str) -> Tuple[torch.Tensor]:
        """
        Retrieves detached representations for a word at the input and
        output of a particular layer module.
        """
        if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
            subtoken = fact_token_strategy[len("subject_") :]
            l_input, l_output = self.get_reprs_at_word_tokens(context_templates, 
                words, layer, module_template, subtoken, "both")
        elif fact_token_strategy == "last":
            raise Exception("This is definitely bugged, fix it.")
        else:
            raise ValueError(f"fact_token={fact_token_strategy} not recognized")

        return l_input.detach(), l_output.detach()


    def get_reprs_at_word_tokens(self, context_templates: List[str], words: List[str], 
            layer: int, module_template: str, subtoken: str, track: str = "in") -> torch.Tensor:
        """Retrieves the last token representation of `word` in `context_template`
        when `word` is substituted into `context_template`. See `get_last_word_idx_in_template`
        for more details."""
        idxs = self.get_words_idxs_in_templates(context_templates, words, subtoken)
        return self.get_reprs_at_idxs(
            [context_templates[i].replace(self.subject_placeholder, words[i]) for i in range(len(words))],
            idxs, layer, module_template, track)


    def get_words_idxs_in_templates(self, context_templates: str, words: str, 
            subtoken: str) -> int:
        """Given list of template strings, each with *one* format specifier
        (e.g. "{} plays basketball"), and words to be substituted into the
        template, computes the post-tokenization index of their last tokens."""
        tok = self.llm.tokenizer
        assert all(
            tmp.count(self.subject_placeholder) == 1 for tmp in context_templates
        ), "We currently do not support multiple fill-ins for context"


        prefixes_len, words_len, suffixes_len, inputs_len = [], [], [], []
        for i, context in enumerate(context_templates):
            prefix, suffix = context.split(self.subject_placeholder)
            prefix_len = len(tok.encode(prefix))
            prompt_len = len(tok.encode(prefix + words[i]))
            input_len = len(tok.encode(prefix + words[i] + suffix))
            prefixes_len.append(prefix_len)
            words_len.append(prompt_len - prefix_len)
            suffixes_len.append(input_len - prompt_len)
            inputs_len.append(input_len)

        # Compute indices of last tokens
        if subtoken == "last" or subtoken == "first_after_last":
            return [
                [
                    prefixes_len[i]
                    + words_len[i]
                    - (1 if subtoken == "last" or suffixes_len[i] == 0 else 0)
                ]
                # If suffix is empty, there is no "first token after the last".
                # So, just return the last token of the word.
                for i in range(len(context_templates))
            ]
        elif subtoken == "first":
            return [[prefixes_len[i] - inputs_len[i]] for i in range(len(context_templates))]
        else:
            raise ValueError(f"Unknown subtoken type: {subtoken}")


    def get_reprs_at_idxs(self, contexts: List[str], idxs: List[List[int]], 
            layer: int, module_template: str, track: str = "in") -> torch.Tensor:
        """Runs input through model and returns averaged representations of the 
            tokens at each index in `idxs`."""
        # contexts: List[str], 
        # idxs: List[List[int]], 
        model = self.llm.model
        tok = self.llm.tokenizer
        def _batch(n):
            for i in range(0, len(contexts), n):
                yield contexts[i : i + n], idxs[i : i + n]

        assert track in {"in", "out", "both"}
        both = track == "both"
        tin, tout = (
            (track == "in" or both),
            (track == "out" or both),
        )#tin tout are both bool
        module_name = module_template.format(layer)
        to_return = {"in": [], "out": []}

        def _process(cur_repr, batch_idxs, key):
            nonlocal to_return
            cur_repr = cur_repr[0] if type(cur_repr) is tuple else cur_repr
            if cur_repr.shape[0]!=len(batch_idxs):
                cur_repr=cur_repr.transpose(0,1)
            for i, idx_list in enumerate(batch_idxs):
                to_return[key].append(cur_repr[i][idx_list].mean(0))

        for batch_contexts, batch_idxs in _batch(n=128):
            #contexts_tok:[21 19]
            contexts_tok = tok(batch_contexts, padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

            with torch.no_grad():
                with nethook.Trace(
                    module=model,
                    layer=module_name,
                    retain_input=tin,
                    retain_output=tout,
                ) as tr:
                    model(**contexts_tok)

            if tin:
                from transformers.models.gptj.modeling_gptj import GPTJForCausalLM
                if isinstance(model, GPTJForCausalLM) and module_name == 'transformer.h.8':
                    with torch.no_grad():
                        with nethook.Trace(
                            module=model,
                            layer=module_name + '.ln_1',
                            retain_input=tin,
                            retain_output=tout,
                        ) as tr2:
                            model(**contexts_tok)
                    tr.input = tr2.input

                _process(tr.input, batch_idxs, "in")
            if tout:
                _process(tr.output, batch_idxs, "out")

        to_return = {k: torch.stack(v, 0) for k, v in to_return.items() if len(v) > 0}

        if len(to_return) == 1:
            return to_return["in"] if tin else to_return["out"]
        else:
            return to_return["in"], to_return["out"]


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )



