from collections import deque
from datetime import datetime
import math
from pathlib import Path
import sys
from tqdm import tqdm
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

from models import CriterionDETR, PrepareInputs, DETR
from datasets import FBPPostProcess


class Engine:
    def __init__(self) -> None:
        self.global_step = 0
        self.writer: Optional[SummaryWriter] = None
        self.loss_window = deque(maxlen=100)

    @staticmethod
    def get_outputs(tokenizer, model, samples, device, attention_words):
        outputs = []
        for doc in samples:
            inputs = tokenizer([doc]).to(device)

            if not len(attention_words):
                enc_attend_every = 10
                times = math.ceil(inputs.size()[1] / enc_attend_every)
                glob_enc_attn = torch.zeros(enc_attend_every, device=device)
                glob_enc_attn[0] = 1
                glob_enc_attn = glob_enc_attn.tile(times)[:inputs.size()[1]]
            else:
                glob_attn_words = [i for i in range(len(inputs[0])) if inputs[0][i] in attention_words]
                glob_enc_attn = torch.zeros(len(inputs[0]), device=device)
                glob_enc_attn[glob_attn_words] = 1
                

            glob_dec_attn = torch.ones(model.num_queries).to(device)

            # BUG Token indices sequence length is longer than the specified maximum 
            # sequence length for this model (16823 > 16384). Running this sequence 
            # through the model will result in indexing errors

            outputs.append(model(inputs, glob_enc_attn, glob_dec_attn))

        batch_outputs = {
            key: torch.cat([o[key] for o in outputs]) for key in outputs[0].keys()
        }

        return batch_outputs

    def train_one_epoch(
        self,
        tokenizer: PrepareInputs,
        model: DETR,
        criterion: CriterionDETR,
        postprocessor: FBPPostProcess,
        data_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch: int,
        max_norm: float = 0,
        attention_words: list = []
    ):
        model.train()
        criterion.train()

        data_bar = tqdm(data_loader, desc=f"Train Epoch {epoch}")
        for samples, targets, infos in data_bar:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            batch_outputs = self.get_outputs(tokenizer, model, samples, device, attention_words)

            loss_dict = criterion(batch_outputs, targets)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)  # type: ignore

            loss_dict_unscaled = {f"{k}_unscaled": v for k, v in loss_dict.items()}
            loss_dict_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict
            }
            losses_scaled = sum(loss_dict_scaled.values())  # type: ignore

            loss_value = losses_scaled.item()  # type: ignore

            postprocessor.add_outputs(batch_outputs, infos)

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict)
                sys.exit(1)

            optimizer.zero_grad()
            losses.backward()  # type: ignore
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)  # type: ignore
            optimizer.step()

            self.loss_window.append(losses.item())  # type: ignore
            data_bar.set_postfix(
                {
                    "loss": f"{sum(self.loss_window) / len(self.loss_window):.3f}",
                    **{k: f"{v.item():.3f}" for k, v in loss_dict_scaled.items()},
                    "lr": ", ".join(f"{g['lr']:.1e}" for g in optimizer.param_groups),
                }
            )
            self.global_step += 1
            if self.writer:
                scalars = {
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": losses.item(),  # type: ignore
                    **loss_dict_scaled,
                    **loss_dict_unscaled,
                }
                for key, value in scalars.items():
                    self.writer.add_scalars(key, {"Train": value}, self.global_step)
        
        report = postprocessor.evaluate()
        if self.writer:
            self.writer.add_scalars("accuracy", {"Train": report["f1"]["macro_avg"]}, self.global_step)
        
        return report


    @torch.no_grad()
    def evaluate(
        self,
        tokenizer: PrepareInputs,
        model: DETR,
        criterion: CriterionDETR,
        postprocessor: FBPPostProcess,
        data_loader: DataLoader,
        epoch: int,
        device: torch.device,
        attention_words: list = []
    ):
        model.eval()
        criterion.eval()

        loss_list = []
        data_bar = tqdm(data_loader, desc=f"Valid Epoch {epoch}")
        for samples, targets, infos in data_bar:
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            batch_outputs = self.get_outputs(tokenizer, model, samples, device, attention_words)

            loss_dict = criterion(batch_outputs, targets)
            weight_dict = criterion.weight_dict

            loss_dict_scaled = {
                k: v * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict
            }
            losses_scaled = sum(loss_dict_scaled.values())

            postprocessor.add_outputs(batch_outputs, infos)

            loss_value = losses_scaled.item()  # type: ignore
            loss_list.append(loss_value)

            data_bar.set_postfix(
                {
                    "loss": f"{sum(loss_list) / len(loss_list):.3f}",
                    **{k: f"{v.item():.3f}" for k, v in loss_dict_scaled.items()},
                }
            )

        loss = sum(loss_list) / len(loss_list)
        report = postprocessor.evaluate()
        scalars = {"loss": loss, "accuracy": report["f1"]["macro_avg"]}

        if self.writer:
            for key, value in scalars.items():
                self.writer.add_scalars(key, {"Validation": value}, self.global_step)

        return report

    def set_outputs(self, args_output_dir) -> Path:
        output_dir = Path(".")
        if args_output_dir:
            timestamp = datetime.now().strftime("%Y_%m_%d-%I_%M")
            output_dir = Path(args_output_dir).joinpath(timestamp)
            self.writer = SummaryWriter(output_dir.joinpath("logs"))
            print("Saving outputs at:", output_dir)
        return output_dir
