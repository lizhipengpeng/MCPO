import os
import random
from typing import Any, Dict, List
import vllm
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#### CONFIG PARAMETERS ---

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 773815))

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 1 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.9 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.


class Phi_ZeroShotModel():
    """
    A dummy model implementation for ShopBench, illustrating how to handle both
    multiple choice and other types of tasks like Ranking, Retrieval, and Named Entity Recognition.
    This model uses a consistent random seed for reproducible results.
    """

    def __init__(self):
        """Initializes the model and sets the random seed for consistency."""
        random.seed(AICROWD_RUN_SEED)
        self.initialize_models()

    def initialize_models(self):
        # Initialize Meta Llama 3 - 8B Instruct Model

        self.model_name = "./models/Phi-3-mini-4k-instruct"
        if not os.path.exists(self.model_name):
            raise Exception(
                f"""
            The evaluators expect the model weights to be checked into the repository,
            but we could not find the model weights at {self.model_name}
            
            Please follow the instructions in the docs below to download and check in the model weights.
                https://gitlab.aicrowd.com/aicrowd/challenges/amazon-kdd-cup-2024/amazon-kdd-cup-2024-starter-kit/-/blob/master/docs/download-baseline-model-weights.md
            
            """
            )

        # initialize the model with vllm
        self.llm = vllm.LLM(
            self.model_name,
            tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="float16", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True,
        )


    def batch_predict(self, prompts) -> List[str]:
        """
        Generates a batch of prediction based on associated prompts and task_type

        For multiple choice tasks, it randomly selects a choice.
        For other tasks, it returns a list of integers as a string,
        representing the model's prediction in a format compatible with task-specific parsers.

        Parameters:
            - batch (Dict[str, Any]): A dictionary containing a batch of input prompts with the following keys
                - prompt (List[str]): a list of input prompts for the model.
    
            - is_multiple_choice bool: A boolean flag indicating if all the items in this batch belong to multiple choice tasks.

        Returns:
            str: A list of predictions for each of the prompts received in the batch.
                    Each prediction is
                           a string representing a single integer[0, 3] for multiple choice tasks,
                        or a string representing a comma separated list of integers for Ranking, Retrieval tasks,
                        or a string representing a comma separated list of named entities for Named Entity Recognition tasks.
                        or a string representing the (unconstrained) generated response for the generation tasks
                        Please refer to parsers.py for more details on how these responses will be parsed by the evaluator.
        """

        
        # set max new tokens to be generated
        max_new_tokens = 50 
        
        # Generate responses via vllm
        responses = self.llm.generate(
            prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0,  # randomness of the sampling
                seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm = False
        )
        # Aggregate answers into List[str]
        batch_response = []
        for response in responses:
            batch_response.append(response.outputs[0].text)        

        return batch_response











