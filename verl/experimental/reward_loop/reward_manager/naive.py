# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import logging
import os

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("naive")
class NaiveRewardManager(RewardManagerBase):
    """The reward manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer

    async def run_single(self, data: DataProto) -> dict:
        data = data[-1:]  # for multi-sequence outputs, we only compute reward based on the last sequence
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        extra_info = dict(data_item.non_tensor_batch.get("extra_info", {}) or {})
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if tool_extra_fields is not None:
            extra_info.update(tool_extra_fields)

        num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
        rollout_reward_scores = data_item.non_tensor_batch.get("reward_scores", {})
        extra_info["num_turns"] = num_turns
        extra_info["rollout_reward_scores"] = rollout_reward_scores

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        extra_info["truncated"] = self._is_truncated(valid_response_ids, response_str)

        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        reward_extra_info = {"truncated": extra_info["truncated"]}

        try:
            if self.is_async_reward_score:
                result = await self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                )
            else:
                result = await self.loop.run_in_executor(
                    None,
                    lambda: self.compute_score(
                        data_source=data_source,
                        solution_str=response_str,
                        ground_truth=ground_truth,
                        extra_info=extra_info,
                        **extra_reward_kwargs,
                    ),
                )

            score: float
            if isinstance(result, dict):
                score = result["score"]
                reward_extra_info.update(result)
            else:
                score = result
                reward_extra_info["acc"] = score
        except Exception as e:
            logger.error(
                f"Reward computation failed for data_source={data_source}: {e}. "
                f"Response preview: {response_str[:100]}..."
            )
            score = 0.0
            reward_extra_info.update(
                {
                    "score": 0.0,
                    "acc": 0.0,
                    "pred": "",
                    "feedback": f"Reward computation failed: {e}",
                    "incorrect_format": 1,
                    "error": str(e),
                }
            )

        reward = score

        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

    def _is_truncated(self, valid_response_ids, response_str: str) -> bool:
        if len(valid_response_ids) == 0:
            return False

        stop_token_ids = {self.tokenizer.eos_token_id}
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id != self.tokenizer.unk_token_id:
            stop_token_ids.add(im_end_id)

        last_token_id = int(valid_response_ids[-1])
        return last_token_id not in stop_token_ids and not response_str.rstrip().endswith("<|im_end|>")
