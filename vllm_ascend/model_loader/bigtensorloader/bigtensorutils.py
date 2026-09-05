import os

from vllm_ascend import envs


class BigTensorsUtils:

    @staticmethod
    def has_safetensors_file() -> bool:
        checkpoint_path = envs.VLLM_ASCEND_CHECKPOINT_PATH
        if not checkpoint_path or not os.path.isdir(checkpoint_path):
            return False
        for filename in os.listdir(checkpoint_path):
            if filename.endswith(".safetensors"):
                return True
            # v2 prototype: post-process snapshot also implies empty placeholders
            # in create_weights (see ops/linear.py, quantization/methods/*)
            if envs.VLLM_ASCEND_BIGTENSOR_V2 and filename.endswith(".v2.snapshot"):
                return True
        return False
