import os
import torch

from transformers import Trainer
from typing import Optional


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


class VTimeLLMTrainer(Trainer):

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Stage1（只训 mm_projector）：全量 checkpoint 会触发 ZeRO-3
            # active_sub_modules 冲突，因此不落全量；改为按步只存
            # mm_projector.bin（约 6MB），中断后可从最近一次保存恢复
            # （把该文件作为 --pretrain_mm_mlp_adapter 重新跑 stage1）。
            # 注意：stage1 checkpoint 不含 trainer_state.json，不会被
            # train.py 的续训 glob 误认成 stage2 断点。
            output_dir = os.path.join(self.args.output_dir,
                                      f"checkpoint-{self.state.global_step}")
            adapter = get_mm_adapter_state_maybe_zero_3(
                self._get_trainable_state_dict(), ['mm_projector'])
            if self.args.local_rank in (-1, 0):
                os.makedirs(output_dir, exist_ok=True)
                torch.save(adapter, os.path.join(output_dir, 'mm_projector.bin'))
                print(f"[stage1] saved mm_projector.bin at step {self.state.global_step}"
                      f" -> {output_dir}")
        else:
            super(VTimeLLMTrainer, self)._save_checkpoint(model, trial)

    def _get_trainable_state_dict(self):
        return self.model.named_parameters()

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # stage1 的最终产物由 train.py 的 safe_save_model_for_hf_trainer
            # 落盘（output_dir/mm_projector.bin），这里保持不动作。
            pass
        else:
            super(VTimeLLMTrainer, self)._save(output_dir, state_dict)
