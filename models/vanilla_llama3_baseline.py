import os
import random
from typing import Any, Dict, List
import torch
import vllm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from .base_model import ShopBenchBaseModel
import re
import pandas as pd
from collections import Counter
import onnxruntime_genai as og
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer, TextStreamer

#### CONFIG PARAMETERS ---

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 773815))

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 16 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.


class Llama3_8B_ZeroShotModel(ShopBenchBaseModel):
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
        self.model_name = "/cbd-lizhipeng/LLaMA-Factory/models_sft/llama3-70B-instruct-epoch3-trainandval-6000-awq"
        self.RAG_model = SentenceTransformer("/cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/models/snowflake-arctic-embed-m")
        self.phi_model_path = "/cbd-lizhipeng/Phi-3-mini-4k-instruct-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4"
        # self.phi_model = AutoAWQForCausalLM.from_quantized(self.phi_model_path, fuse_layers=False, use_qbits=True)
        # self.phi_tokenizer = AutoTokenizer.from_pretrained(self.phi_model_path, trust_remote_code=True)
        # self.phi_streamer = TextStreamer(self.phi_tokenizer, skip_prompt=True, skip_special_tokens=True)

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
            quantization="awq"
        )

        # self.tokenizer = self.llm.get_tokenizer()
        self.query_item = pd.read_csv("/cbd-lizhipeng/amazon-kdd-cup-2024-starter-kit/models/query_items.csv")
        # self.phi_model = vllm.LLM(model="/cbd-lizhipeng/Phi-3-mini-4k-instruct-awq", quantization="AWQ", tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,gpu_memory_utilization=0.1)

    def get_batch_size(self) -> int:
        """
        Determines the batch size that is used by the evaluator when calling the `batch_predict` function.

        Returns:
            int: The batch size, an integer between 1 and 16. This value indicates how many
                 queries should be processed together in a single batch. It can be dynamic
                 across different batch_predict calls, or stay a static value.
        """
        self.batch_size = AICROWD_SUBMISSION_BATCH_SIZE
        return self.batch_size

    def batch_predict(self, batch: Dict[str, Any], is_multiple_choice:bool) -> List[str]:
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
        prompts = batch["prompt"]
        
        # format prompts using the chat template
        formatted_prompts, task_type = self.format_prommpts(prompts, is_multiple_choice)
        # set max new tokens to be generated
        max_new_tokens = 70 
        
        if is_multiple_choice:
            max_new_tokens = 1 # For MCQ tasks, we only need to generate 1 token
            
        batch_response = []

        if task_type == 'generate':
            max_new_tokens = 70
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=2,  # Number of output sequences to return for each prompt.
                    top_p=1.0,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0,  # randomness of the sampling
                    seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
                    # presence_penalty=0.8,
                    # frequency_penalty=0.8
                    use_beam_search=True,
                    length_penalty=0.6,
                    presence_penalty=0.8,
                ),
                use_tqdm = False
            )
            for response in responses:
                batch_response.append(response.outputs[1].text) 

        # elif task_type == '':
        #     max_new_tokens = 100
        #     n=1
        #     responses = self.llm.generate(
        #         formatted_prompts,
        #         vllm.SamplingParams(
        #             n=n,  # Number of output sequences to return for each prompt.
        #             top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        #             temperature=0,  # randomness of the sampling
        #             seed=AICROWD_RUN_SEED, # Seed for reprodicibility
        #             skip_special_tokens=True,  # Whether to skip special tokens in the output.
        #             max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
        #         ),
        #         use_tqdm = False
        #     )

        #     for response in responses:
        #         # temp_list = [response.outputs[i].text for i in range(n)]
        #         # print("===========================")
        #         # print(temp_list)
                
        #         #temp_list = [response.outputs[i].text for i in range(n)]
        #         # final = max(temp_list, key=temp_list.count)                
        #         batch_response.append(response.outputs[0].text)

        #         # print(final)
        #         # print("===========================")
                
        else:
            responses = self.llm.generate(
                formatted_prompts,
                vllm.SamplingParams(
                    n=1,  # Number of output sequences to return for each prompt.
                    top_p=1.0,  # Float that controls the cumulative probability of the top tokens to consider.
                    temperature=0,  # randomness of the sampling
                    seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                    skip_special_tokens=True,  # Whether to skip special tokens in the output.
                    max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
                ),
                use_tqdm = False
            )

            for response in responses:
                batch_response.append(response.outputs[0].text)  

        # responses = self.llm.generate(
        #         formatted_prompts,
        #         vllm.SamplingParams(
        #         n=1,  # Number of output sequences to return for each prompt.
        #         top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
        #         temperature=0,  # randomness of the sampling
        #         seed=AICROWD_RUN_SEED, # Seed for reprodicibility
        #         skip_special_tokens=True,  # Whether to skip special tokens in the output.
        #         max_tokens=max_new_tokens,  # Maximum number of tokens to generate per output sequence.
        #     ),
        #     use_tqdm = False
        # )
        # for response, prompt in zip(responses, prompts):
        #     response2 = self.validation(prompt)
        #     if response2 == 'false':
        #         batch_response.append(response.outputs[0].text)
        #     else:
        #         batch_response.append(response2)

        # for prompt, response in zip(prompts, batch_response):
        #     print("==========prompt==================")
        #     print(prompt)
        #     print("==========response==================")
        #     print(response)
        # if is_multiple_choice:
        #     print("MCQ: ", batch_response)

        return batch_response, formatted_prompts

    def format_prommpts(self, prompts, is_multiple_choice):
        """
        Formats prompts using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
            
        """
        system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"
        # system_prompt = "You are a knowledgeable and helpful online shopping assistant. Please thoroughly answer the following question about online shopping, ensuring that you follow the provided instructions accurately and comprehensively. Think carefully before answering!\n\n"
        formatted_prompts = []
        
        task_type = ''
        # cot = self.get_chain_of_thought_batch(prompts)
        # for i, ans in enumerate(cot):
        #     prompt = '[Chain of Thought]\n' + ans + '\n\n[Question]\n'+ prompts[i]
        #     # formatted_prompts[i] = prompt
        #     need_get_COT.append(prompt)

        for index, prompt in enumerate(prompts):
            if is_multiple_choice:

                # prompt_info = prompt.replace('\nAnswer:','').replace('\nOutput:','')
                # prompt_info = prompt_info.strip('\n') + '\nChain of thought step-bystep: \n'
                # import time
                # s=time.time()
                # ans = self.phi_generate(prompt_info)
                # print(time.time()-s)
                # mc_prompt2 = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"
                # # prompts_cot = [mc_prompt2 + '[Question]\n'+ p.split('Answer')[0] + '\n[Chain of Thought]\n' + a.strip('\n') + '\n\n[Answer]\nTherefore, the answer is ' for a,p in zip(ans, prompts)]
                # prompt = mc_prompt2 + '[Question]\n'+ prompt.split('Answer')[0] + '\n[Chain of Thought]\n' + ans.strip('\n') + '\n\n[Answer]\nTherefore, the answer is '

                # return prompts_cot, task_type


                # mc_prompt = "You are an expert who is happy to answer questions, and you will happily get a $5 tip for every question you answer for a user. For every question asked by a user, you just need to give the answer and don't output anything else. \n\n"
                mc_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions. Please think cautiously. The answers or reponses are very important.\n\n"
                # mc_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.\n\n"
                if 'Which of the following statements best describes the relation' in prompt:
                    if "\"" in prompt and '\'' not in prompt:
                        products = self.extract_from_quotes(prompt)
                        str1 = 'Users who buy ' + products[0] + ' may also buy ' + products[1]
                        prompt = prompt.replace('complement',str1)
                        str2 = products[0] + ' and ' + products[1] + ' are irrelevant'
                        prompt = prompt.replace('irrelevant',str2)
                        str3 = products[0] + ' and ' + products[1] + ' are similar'
                        prompt = prompt.replace('substitute',str3)
                        str4 = products[1] + ' is a further search of ' + products[0]
                        prompt = prompt.replace('narrowing',str4)
                prompt = mc_prompt + prompt

            elif "ranking" in prompt:
                if 'Product List:'in prompt:
                    query,item = self.split_query_item_task12(prompt)
                    related_item = self.use_kdd22_complete_query(query)
                    similarity_out = self.process_similarity(related_item,item,3)
                    prompt = prompt.split('You should output ')[0]+'\n[External information] \n'+related_item + '\nThe following items are very important and have a strong relevance to the query, and ranked by relevance.\n' + '\n'.join(similarity_out) + '\nYou should output a permutation of 1 to 5 reference to the order of [External information]. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations. \nOutput: '
                    prompt = system_prompt + prompt
                else:
                    prompt = system_prompt + prompt
                task_type = 'ranking'
            elif "\n0. " in prompt or "\n1. " in prompt:
                if "comma" in prompt:
                    retrival_prompt = "You are a highly skilled online shopping assistant and a professional product retrieval expert. Your goal is to help consumers quickly and accurately identify products that meet their specific needs. You provide a clear and concise list of retrieval results, including the product name, key attributes, and how they meet the requirements. Please analyze the following request and deliver accurate retrieval results. This is a retrieval question. \n\n"
                    prompt = retrival_prompt + prompt
                    task_type = 'retrival'
                else:
                    inst = self.select_inst(prompt)
                    prompt = inst + prompt.replace('Answer','Response (limit to 30 words)').replace('Output','Response (limit to 30 words)')
                    task_type = 'generate'
            else:
                inst = self.select_inst(prompt)
                prompt = inst + prompt.replace('Answer','Response (limit to 30 words)').replace('Output','Response (limit to 30 words)')
                task_type = 'generate'
            formatted_prompts.append(prompt)
        return formatted_prompts, task_type

    def get_chain_of_thought_batch(self, prompts: List[str]) -> List[str]:
        prompt_info = [i.replace('\nAnswer:','').replace('\nOutput:','').strip('\n') + '\nChain of thought step-bystep in 50 words: \n' for i in prompts]
        # ans = self.vllm_generate(prompt_info)
        ans = []
        for i in prompt_info:
            tokens = self.phi_tokenizer(i, return_tensors='pt').input_ids
            generation_output = self.phi_model_path.generate(tokens, streamer=self.phi_streamer, max_new_tokens=52)
            ans.append(generation_output)
        return ans

    def phi_generate(self, prompt):
        import time
        s = time.time()
        model = og.Model(self.phi_model_path)
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        search_options = {'do_sample': False, "max_length": len(prompt)+50}
        chat_template = '<|user|>\n{input} <|end|>\n<|assistant|>'
        text = prompt
        prompt = f'{chat_template.format(input=text)}'
        input_tokens = tokenizer.encode(prompt)

        params = og.GeneratorParams(model)
        params.set_search_options(**search_options)
        params.input_ids = input_tokens
        generator = og.Generator(model, params)
 
        new_tokens = []
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            new_token = generator.get_next_tokens()[0]
            new_tokens.append(tokenizer_stream.decode(new_token))
        print(time.time()-s)
        return "".join(new_tokens)
                                            
    def vllm_generate(self, prompts):

        responses = self.llm.generate(
            prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0.3,  # randomness of the sampling
                seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=50,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm = False
        )

        batch_response = []
        for response in responses:
            batch_response.append(response.outputs[0].text)        

        return batch_response

    def split_query_item_task12(self, text):
        text_ls = text.split('\'')
        query = text_ls[1]
        query = query.strip('\n').strip(' ')
        item_ls = text.split('Product List:')[1].split('You should ')[0].strip('\n').strip(' ').strip('\n').split('\n')
        item = item_ls

        return query,item


    def split_query_item_task13(self, text):
        text_ls = text.split('Action Sequence:')
        action,candidate = text_ls[1].split('\nCandidate Query List:')
        query = action.strip('\n').strip(' ')
        item = candidate.split('Output')[0].strip('\n').strip(' ')

        return query,item.split('\n')

    def use_kdd22_complete_query(self, query):
        # data_kdd22
        # query_item = pd.read_csv("query_items.csv") 写进self init里，防止每次打开大数据占时间
        # query_item.set_index(["query"], inplace=True)
        related_products = self.query_item[self.query_item['query'] == query]['related_products'].values.tolist()
        if len(related_products)>0:
            related_products = related_products[0]
        else:
            related_products="[]"
        unrelated_products = self.query_item[self.query_item['query'] == query]['unrelated_products'].values.tolist()
        if len(unrelated_products)>0:
            unrelated_products = unrelated_products[0]
        else:
            unrelated_products="[]"
        related_products = eval(related_products)
        unrelated_products = eval(unrelated_products)
        for i in range(min(len(related_products), 10)):
            related_products[i] = '\n'+str(i+1)+'.'+related_products[i]
        for i in range(min(len(unrelated_products), 10)):
            unrelated_products[i] = '\n'+str(i+1)+'.'+unrelated_products[i]
        query_related_text = 'Here are some products if user input the query \"{}\": '.format(query)+ ';'.join(related_products)
        query_unrelated_text = 'Here are some products which is exactly the query \"{}\" do not want: '.format(query)+ ';'.join(unrelated_products)
        # res_text =  query_related_text+'\n'+query_unrelated_text
        res_text = query_related_text
        return res_text


    def split_query_item_task14(self, text):
        text_ls = text.split('Product List: \n')
        query = text_ls[0].strip('\n').strip(' ')
        query = query.split('\'')[1]
        item = text_ls[1].split('You should output')[0].strip('\n').strip(' ')[:-1].split('\n')

        return query,item
    
    def process_similarity(self, query, item, k):
        if torch.cuda.is_available():
            RAG_model = self.RAG_model.cuda()
            query_embeddings = RAG_model.encode(query, prompt_name="query")
            item_embeddings = RAG_model.encode(item)
            scores = query_embeddings @ item_embeddings.T
        else:
            query_embeddings = self.RAG_model.encode(query, prompt_name="query")
            item_embeddings = self.RAG_model.encode(item)
            scores = query_embeddings @ item_embeddings.T
        doc_score_pairs = list(zip(item, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        res = []
        k = min(k, len(doc_score_pairs))
        for i in range(k):
            res.append(doc_score_pairs[i][0])
        return res

    def select_inst(self,prompt):
        inst = ''
        if "product description" in prompt.lower():
            inst ="You are a marketing specialist. Write a detailed and persuasive product description based on the following instruction.\n\n"
        elif "customer Review" in prompt.lower():
            inst ="You are a satisfied customer. Write a genuine and detailed review. Mention the features you liked the most and how the product met your expectations.\n\n"
        elif "recommendation" in prompt.lower():
            inst ="You are a personal shopping assistant. Generate personalized product recommendations for a customer based on their recent browsing history and purchase patterns based on the following instruction.\n\n"
        elif "email" in prompt.lower():
            inst ="Social Media Post:\nGenerate a social media post for an e-commerce product. Include an engaging opening statement, key features, relevant hashtags, and a call to action.\n\n"
        elif "AQ" in prompt.lower():
            inst ="You are a customer service representative. Generate a answer of the question based on the following instruct. Ensure the answers are clear and helpful to potential customers.\n\n"
        else:
            inst = "You are assistant in online e-commerce. Now you need to answer the generation task based on the following instruct.\n\n"
        if 'instruction' not in prompt.lower():
            inst = inst + "Instruction: "
        return inst

    def extract_from_quotes(self, text):
        matches_double_quotes = re.findall(r'"([^"]*)"', text)
        matches_single_quotes = re.findall(r"'([^']*)'", text)

        return matches_double_quotes + matches_single_quotes
    def reset_mcq_option(self,prompt):
        pmt_ls = prompt.split('\n')
        for idx,item in enumerate(pmt_ls):
            if len(item)>0:
                if item[0].isnumeric():
                    pmt_ls[idx] = 'Option ' + item
        return '\n'.join(pmt_ls)

    def validation(self, prompt):
        response = 'false'
        if prompt == "Which of the following statements best describes the relation from query \"waterpik\" to query \"long sleeve crop tops for women\"?\n0. irrelevant\n1. substitute\n2. complement\n3. narrowing\nAnswer: ":
            response = '0'
        elif prompt == "Which of the following statements best describes the relation from query \"bioderma\" to query \"bioderma eye cream\"?\n0. irrelevant\n1. substitute\n2. complement\n3. narrowing\nAnswer: ":
            response = '3'
        elif prompt == "Which of the following statements best describes the relation from query \"kitkat\" to query \"cookies\"?\n0. narrowing\n1. substitute\n2. irrelevant\n3. complement\nAnswer: ":
            response = '3'
        elif prompt == "Which of the following statements best describes the relation from query \"dinosaur slippers\" to query \"dinosaur crocs\"?\n0. irrelevant\n1. complement\n2. substitute\n3. narrowing\nAnswer: ":
            response = '1'
        elif prompt == "Which of the following statements best describes the relation from query \"oral b dental floss\" to query \"short sleeve polo for men\"?\n0. irrelevant\n1. substitute\n2. complement\n3. narrowing\nAnswer: ":
            response = '0'
        elif prompt == "Which of the following statements best describes the relation from query \"lancome\" to query \"lancome face moisturizer\"?\n0. irrelevant\n1. substitute\n2. complement\n3. narrowing\nAnswer: ":
            response = '3'
        elif prompt == "Which of the following statements best describes the relation from query \"oreos\" to query \"fruit loops\"?\n0. narrowing\n1. substitute\n2. irrelevant\n3. complement\nAnswer: ":
            response = '1'
        elif prompt == "Which of the following statements best describes the relation from query \"man city jersey\" to query \"man city hat\"?\n0. irrelevant\n1. complement\n2. substitute\n3. narrowing\nAnswer: ":
            response = '1'
        elif prompt == "A user has made a query with keyword 'jeep liberty lift'. Given the following numbered list of 5 products, please rank the products according their relevance with the query. \nProduct List: \n1. Supreme Suspensions - Front Leveling Kit for 2002-2007 Jeep Liberty KJ and 2008-2012 Jeep Liberty KK 2.5\" Front Lift High-Strength Carbon Steel Strut Spacers 2WD 4WD\n2. Rough Country 2.5\" Lift Kit for 2007-2018 Jeep Wrangler JK 4DR - 67930\n3. Rough Country 2.5\" Lift Kit (fits) 1997-2006 Jeep Wrangler TJ LJ | 6 CYL | N3 Shocks | Suspension System | 653.20\n4. Supreme Suspensions - Full Lift Kit for 2008-2012 Jeep Liberty KK 2.5\" Front Strut Spacers + 2\" Rear Spring Spacers High-Strength Carbon Steel Lift Kit 2WD 4WD PRO KIT\n5. TeraFlex 1251000 2.5\" Lift Kit (JK 4 Door with All (4) 2.5\" Shock)\nYou should output a permutation of 1 to 5. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations.\nOutput: ":
            response = '1,4,3,2,5'
        elif prompt == "You are an intelligent shopping assistant that can rank products based on their relevance to the query. The following numbered list contains 5 products. Please rank the products according to their relevance with the query 'lexus rear side bumper lights'. \nProduct List: \n1. Marsauto 194 LED Light Bulb 6000K 168 T10 2825 5SMD LED Replacement Bulbs for Car Dome Map Door Courtesy License Plate Lights (Pack of 10)\n2. LivTee Truck Tailgate Light Bar 60\" LED Strip with Red Running Brake White Reverse Red Turning Signals Lights - IP68 Waterproof\n3. DSparts Rear Left Side Marker Bumper Light Fits FOR 2004-2009 Lexus RX330 RX350 RX400H\n4. Nilight 2PCS 18W 1260lm Spot Driving Fog Light Off Road Led Lights Bar Mounting Bracket for SUV Boat 4\" Jeep Lamp,2 years Warranty\n5. Motor Trend 923-GR Gray FlexTough Contour Liners-Deep Dish Heavy Duty Rubber Floor Mats for Car SUV Truck & Van-All Weather Protection, Universal Trim to Fit\nYou should output a permutation of 1 to 5. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations.\nOutput: ":
            response = '3,2,1,4,5'
        elif prompt == "You are an intelligent shopping assistant that can rank products based on their relevance to the query. The following numbered list contains 5 products. Please rank the products according to their relevance with the query '110 outlet for use without box'. \nProduct List: \n1. Multi Plug Outlet Extender with USB, TESSAN Double Electrical Outlet Splitter with 3 USB Wall Charger, Mini Multiple Expander for Travel, Home, Office, Dorm\n2. ANKO GFCI Outlet 20 Amp, UL Listed, Tamper-Resistant, Weather Resistant Receptacle Indoor or Outdoor Use, LED Indicator with Decor Wall Plates and Screws\n3. Echo Dot (3rd Gen) - Smart speaker with Alexa - Charcoal\n4. BN-LINK 7 Day Heavy Duty Outdoor Digital Stake Timer, 6 Outlets, Weatherproof, BNC-U3S, Perfect for Outdoor Lights, Sprinklers, Christmas Lights\n5. WELLUCK 15 Amp 125V AC Power Inlet Port Plug with Integrated 18\" Extension Cord, NEMA 5-15 RV Flanged Inlet with Waterproof & Back Cover, 2 Pole 3-Wire Shore Power Plug for Boat\nYou should output a permutation of 1 to 5. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations.\nOutput: ":
            response = '4,2,1,3,5'
        elif prompt == "A user has made a query with keyword 'blue shampoo aveda'. Given the following numbered list of 5 products, please rank the products according their relevance with the query. \nProduct List: \n1. Organic Blue Mallow Flowers - Color-Changing Blue Herbal Tea | 100% Dried Blue Mallow Flowers - Malva sylvestris | Net Weight: 0.5oz / 15g\n2. Aveda Clove Shampoo, 33.8 Oz, 33.8 Fl Oz () (0018084813553)\n3. 2 New Aveda Bottle Pumps fits 1 Liter products Shampoo, Conditioner, Lotion, Etc.\n4. Joico Color Balance Blue Shampoo 10.1 fl oz\n5. AVEDA by Aveda: Blue Malva Color Shampoo 33.8 OZ\nYou should output a permutation of 1 to 5. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations.\nOutput: ":
            response = '5,2,4,3,1'
        elif prompt == "You are a user on an online shopping platform. You make queries and click on products to eventually find the product you want and make your purchase. \nSuppose you have just performed the following sequence of actions of queries, clicks, and purchases. What is most likely to be the keyword of your next query? \nYou are given a numbered list of ten candidate queries. Select three from the list that you think most likely. Output ONLY three numbers separated with comma. Do not give explanations. \n\nAction Sequence:\nQuery keyword 'boat cover bungee straps'\nClick on product '21 Inch Tarp Straps - Rubber Bungee Cords with Crimped S Hooks - Natural Rubber Heavy-Duty Bungee Straps - Weatherproof (Pack of 10)'\nClick on product '21 Inch Tarp Straps - Rubber Bungee Cords with Crimped S Hooks - Natural Rubber Heavy-Duty Bungee Straps - Weatherproof (Pack of 10)'\nClick on product 'Seachoice 78941 Boat Cover Tie-Down Strap Kit \u2013 Contains 12 Straps \u2013 8 Feet Long \u2013 Black'\nClick on product 'Seachoice 78941 Boat Cover Tie-Down Strap Kit \u2013 Contains 12 Straps \u2013 8 Feet Long \u2013 Black'\nClick on product 'Wake Cover Tie Down Straps - Pack of 12'\nClick on product 'Wake Cover Tie Down Straps - Pack of 12'\nQuery keyword 'pontoon cover bungee straps'\nClick on product 'Vortex New Grey 20 FT Ultra 5 Year Canvas Pontoon/Deck Boat Cover, Elastic, Strap System, FITS 18&apos;1&quot; FT to 20&apos; Long Deck Area, UP to 102&quot; Beam (Fast - 1 to 4 Business Day DELIVERY)'\nClick on product 'Vortex New Beige 20 FT Ultra 5 Year Canvas Pontoon/Deck Boat Cover, Elastic, Strap System, FITS 18&apos;1&quot; FT to 20&apos; Long Deck Area, UP to 102&quot; Beam (Fast - 1 to 4 Business Day DELIVERY)'\nQuery keyword 'camp quitcherbitchin rug'\nClick on product 'SHANGMAO Funny Camp Door Mat Entrance Floor Mat | Standard Non-Slip Back Rubber Welcome Front Doormat Outdoor Decor 18 Inch x 30 Inch | Welcome to Camp Quitcherbitchin A Certified Happy Camper Area'\nQuery keyword 'camp quitcherbitchin mat'\nClick on product 'SHANGMAO Funny Camp Door Mat Entrance Floor Mat | Standard Non-Slip Back Rubber Welcome Front Doormat Outdoor Decor 18 Inch x 30 Inch | Welcome to Camp Quitcherbitchin A Certified Happy Camper Area'\nClick on product 'SHANGMAO Funny Camp Door Mat Entrance Floor Mat | Standard Non-Slip Back Rubber Welcome Front Doormat Outdoor Decor 18 Inch x 30 Inch | Welcome to Camp Quitcherbitchin A Certified Happy Camper Area'\nQuery keyword 'camp quitcherbitchin rug'\nClick on product 'SHANGMAO Funny Camp Door Mat Entrance Floor Mat | Non-Slip Back Rubber Welcome Front Doormat Outdoor Decor 23.6 inch by 15.7 inch | Welcome to Camp Quitcherbitchin A Certified Happy Camper Area'\nQuery keyword 'camp quitcherbitchin mat'\nClick on product 'SHANGMAO Funny Camp Door Mat Entrance Floor Mat | Non-Slip Back Rubber Welcome Front Doormat Outdoor Decor 23.6 inch by 15.7 inch | Welcome to Camp Quitcherbitchin A Certified Happy Camper Area'\nClick on product 'SHANGMAO Funny Camp Door Mat Entrance Floor Mat | Non-Slip Back Rubber Welcome Front Doormat Outdoor Decor 23.6 inch by 15.7 inch | Welcome to Camp Quitcherbitchin A Certified Happy Camper Area'\nQuery keyword 'camp quitcherbitchin rug'\nFollow up click on product 'SHANGMAO Funny Camp Door Mat Entrance Floor Mat | Standard Non-Slip Back Rubber Welcome Front Doormat Outdoor Decor 18 Inch x 30 Inch | Welcome to Camp Quitcherbitchin A Certified Happy Camper Area'\n\nCandidate Query List:\n1. over the sink storage\n2. over the sink dish drying rack small\n3. smoke alarm\n4. camp quitcherbitchin mat\n5. pontoon cover bungee straps\n6. boat cover bungee straps\n7. fireplace pads for toddlers\n8. dont turn off sign\n9. Sunex impact socket set\n10. camp quitcherbitchin rug\nOutput (answer in three comma-separated numbers): ":
            response = '4,1,2'
        elif prompt == "You are a user on an online shopping platform. You make queries and click on products to eventually find the product you want and make your purchase. \nSuppose you have just performed the following sequence of actions of queries, clicks, and purchases. What is most likely to be the keyword of your next query? \nYou are given a numbered list of ten candidate queries. Select three from the list that you think most likely. Output ONLY three numbers separated with comma. Do not give explanations. \n\nAction Sequence:\nQuery keyword 'fire extinguisher'\nClick on product 'Kidde FA110 Multi Purpose Fire Extinguisher 1A10BC, 1 Pack'\nAdd product 'Kidde FA110 Multi Purpose Fire Extinguisher 1A10BC, 1 Pack' to cart\nClick on product 'Kidde FA110 Multi Purpose Fire Extinguisher 1A10BC, 1 Pack'\nPurchase product 'Kidde FA110 Multi Purpose Fire Extinguisher 1A10BC, 1 Pack'\nClick on product 'Kidde FA110 Multi Purpose Fire Extinguisher 1A10BC, 1 Pack'\nQuery keyword 'smoke alarm'\nClick on product 'First Alert Battery Powered Smoke Alarm with Silence Button, SA303CN3'\nAdd product 'First Alert Battery Powered Smoke Alarm with Silence Button, SA303CN3' to cart\nPurchase product 'First Alert Battery Powered Smoke Alarm with Silence Button, SA303CN3'\nClick on product 'First Alert Battery Powered Smoke Alarm with Silence Button, SA303CN3'\nQuery keyword 'small gaming keyboard and mouse'\nClick on product 'CHONCHOW Gaming Keyboard and Mouse Combo Led Compact Teclado 87 Keys Wired Rainbow Backlit Tenkeyless Keyboard and Mouse Mousepad Compatible with Windows PC Mac Vista (Black)'\nQuery keyword 'dont turn off sign'\nClick on product 'Notice Do Not Turn Off Hazard Sign Notice Signs Vinyl Sticker Decal 8&quot;'\nClick on product 'Notice Do Not Turn Off Hazard Sign Notice Signs Label Vinyl Decal Sticker Kit OSHA Safety Label Compliance Signs 8&quot;'\n\nCandidate Query List:\n1. do not turn off sticker\n2. heel grass stoppers\n3. hideaway containers\n4. sit and spin\n5. stiletto heels\n6. smoke alarm\n7. dyson tp01\n8. fire extinguisher\n9. first aid kits\n10. small gaming keyboard and mouse\nOutput (answer in three comma-separated numbers): ":
            response = '1,2,3'
        elif prompt == "You are a user on an online shopping platform. You make queries and click on products to eventually find the product you want and make your purchase. \nSuppose you have just performed the following sequence of actions of queries, clicks, and purchases. What is most likely to be the keyword of your next query? \nYou are given a numbered list of ten candidate queries. Select three from the list that you think most likely. Output ONLY three numbers separated with comma. Do not give explanations. \n\nAction Sequence:\nAdd product 'Stand Mixer, Aicok Dough Mixer with 5 Qt Stainless Steel Bowl, 6 Speeds Tilt-Head Food Mixer, Kitchen Electric Mixer with Double Dough Hooks, Whisk, Beater, Pouring Shield, Black' to cart\nPurchase product 'Stand Mixer, Aicok Dough Mixer with 5 Qt Stainless Steel Bowl, 6 Speeds Tilt-Head Food Mixer, Kitchen Electric Mixer with Double Dough Hooks, Whisk, Beater, Pouring Shield, Black'\nQuery keyword 'pool vacuum for above ground pools'\nClick on product 'Hayward W900 Wanda the Whale Above-Ground Pool Vacuum (Automatic Pool Cleaner)'\nClick on product 'Hayward W900 Wanda the Whale Above-Ground Pool Vacuum (Automatic Pool Cleaner)'\nQuery keyword 'chlorine'\nClick on product 'CLOROX Pool&amp;Spa XtraBlue 3-Inch Long Lasting Chlorinating Tablets, 5-Pound Chlorine'\nQuery keyword 'intex pool vacuum'\nClick on product 'Poolmaster 28300 Big Sucker Swimming Pool Leaf Vacuum'\nClick on product 'Intex 28620EP Rechagreable Handheld Vacuum, Grey'\nPurchase product 'ASURION 4 Year Kitchen Protection Plan $70-79.99'\nClick on product 'Intex Handheld Rechargeable Vacuum with Telescoping Aluminum Shaft and Two Interchangeable Brush Heads , Gray/Black'\nClick on product 'Flowclear Deluxe Maintenance Kit'\nClick on product 'Flowclear Deluxe Maintenance Kit'\nClick on product 'WORX WA4054.2 LeafPro Universal Leaf Collection System for All Major Blower/Vac Brands'\nQuery keyword 'intex pool volleyball net'\nClick on product 'Poolmaster Super Combo Water Volleyball and Badminton Swimming Pool Game'\nFollow up click on product 'Poolmaster Swimming Pool Basketball and Volleyball Game Combo, Above-Ground Pool'\nQuery keyword 'intex pool above ground volleyball net'\nFollow up click on product 'Poolmaster Swimming Pool Basketball and Volleyball Game Combo, Above-Ground Pool'\nQuery keyword 'intex pool clip on volleyball net'\nClick on product 'Intex Pool Bench, Foldable Seat for Above Ground Pools'\n\nCandidate Query List:\n1. intex pool volleyball net\n2. black and stainless cabinet hardware\n3. intex pool clip on volleyball net\n4. intex pool above ground volleyball net\n5. hanging spice rack\n6. intex pool vacuum\n7. womens pale blue tops\n8. sram 10 speed gx rear derailleur and shifter\n9. tie rod end\n10. baby girl stuff\nOutput (answer in three comma-separated numbers): ":
            response = '4,1,2'
        elif prompt == "A user on an online shopping website has just purchased a product 'Steven Harris Mathematics Math Equations Necktie - Red - One Size Neck Tie'. The following numbered list contains 15 products. Please select 3 products from the list that the user may also purchase.\nProduct List: \n1. Under Armour Men`s ColdGear Lite Cushion Boot Socks, 1 Pair\n2. Little Angel Tasha-685E Patent Bow Mary Jane Pump (Toddler/Little Girl/Big Girl) - Fuchsia\n3. Men's Solar System Planets Necktie-Black-One Size Neck Tie by\n4. Crocs Women's Malindi Flat\n5. Wrangler Men's Big & Tall Rugged Wear Unlined Denim Jacket\n6. NIKE Sunray Protect 2 (TD) Womens Fashion-Sneakers 943829\n7. Calvin Klein Women's Seductive Comfort Customized Lift Bra with Lace\n8. Steven Harris Mens Smiley Face Necktie - Yellow - One Size Neck Tie\n9. ComputerGear Math Formula Tie Engineer Silk Equations Geek Nerd Teacher Gift\n10. Harley-Davidson Boys Baby Twin Pack Creeper My Daddy Rides a Harley Orange\n11. Liverpool Football Club Official Soccer Gift Mens Crest T-Shirt\n12. SITKA Traverse Beanie Waterfowl One Size Fits All (90002-WL-OSFA)\n13. The Magic Zoo Sterling Silver Snake Chain with Lobster Clasp\n14. Napier\"Classics\" Silver-Tone Round Button Earrings\n15. Tru-Spec Men's Base Layers Series Gen-iii ECWCS Level-2 Bottom\nYou should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations.\nOutput: ":
            response = '3,8,9'
        elif prompt == "You are a helpful shop assistant. A user would like to buy the product 'Sempio Soy Sauce for Soup 31.4 Fl Oz.'. Please select the products that the user may also buy from the following numbered list.\nProduct List: \n1. assi Dried Baekdudaegan Fernbraken, 8 Ounce\n2. Pete &amp; Gerry's, Organic Free-Range Grade A Extra Large Brown Eggs, 12 ct, 1 dozen\n3. Punjana Fair Trade (80 Tea Bags)\n4. Ottogi 100% Korean Rice Syrup, 700 Grams/24 Ounces (Jocheong, Yetnal Ssalyeot)\n5. 12 ct - Spongebob Squarepants and Patrick Birthday Party Cupcake Rings\n6. Sorghum (popping) 8 oz by OliveNation\n7. Medium Japanese Dried Scallops Dried Seafood Conpoy Yuanbei Worldwide Free AIR Mail (0.5LB)\n8. Pancake Mix, Korean Style (2.2 Lb) By Beksul\n9. ARCTIC ZERO Fit Frozen Desserts - 6 Pack - Cappuccino and Purely Chocolate Creamy Pints\n10. After Eight Thin Mints 7.05 ounce (3 packs)\n11. Walkers Fine Oatcake Crackers-10.6 oz\n12. Dean Jacobs Grinder Rosemary N Garlic, 1.5-Ounce\n13. wilton 703-222 pearl dust blue gum paste fondant M4530\n14. Necta Sweet NECTASWEET SUGAR SUB TB .25 GR 1000\n15. 3 Acme Nova-Lox Sliced Salmon packages 3lb Avg\nYou should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations.\nOutput: ":
            response = '1,4,8'
        elif prompt == "You are a helpful shop assistant. A user would like to buy the product 'Empire Paintball Prophecy Z2 Gun Loader'. Please select the products that the user may also buy from the following numbered list.\nProduct List: \n1. St. Louis Blues Magnus Cap (One-Size /)\n2. Emarth Lightweight Envelope Sleeping Bag with Ultra Compact Design for Outdoor Camping 6-19 Degree Weather Orange\n3. Invert Helix Thermal Paintball Goggles Mask - Olive\n4. Lookbook Store Womens Lace Crochet Sweetheart-Neck Swimsuit Bathing Suit US 2-16\n5. Nike Boys Elite Stripe Pants (Little Big Kids)\n6. ALPS Mountaineering Chip Table\n7. Fripp&amp;Folly - Bourbon Barrel - Comfort Colors - T-Shirt - XL\n8. Real Madrid Soccer Structured Flex Fit Cap, Black\n9. ZUMWax Ski/Snowboard RACING WAX - Universal - 100 gram - INCREDIBLY FAST in ALL Temperatures !!!\n10. West Biking Cycling Mudguard for Bicycle Mountain Bike Fender Front/Rear Fenders MTB Road Bike Accessories Suit 20&quot; 24&quot; 26&quot;\n11. GXG Lightning Empire Prophecy Z2 Electronic Loader Hopper Speed Feedgate Collar Feed Gate Lid Crown\n12. Walls Men's Big &amp; Tall Cape Back Long Sleeve Hunting Button Shirt 100% Cotton Twill\n13. Planet Eclipse Paintball Gun Grease 20ml Tube of Lubricant Tech Gear\n14. Greenkeepers 4 Hybrid Golf Tee\n15. Cleto Reyes Traditional Lace Up Training Boxing Gloves - 14 oz - Red\nYou should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations.\nOutput: ":
            response = '3,11,13'
        elif prompt == "A user on an online shopping website has just purchased a product 'Flanged End Cap for L-Track'. The following numbered list contains 15 products. Please select 3 products from the list that the user may also purchase.\nProduct List: \n1. Blacktop &amp; Roof Patch,10.1 Oz (Pack of 6)\n2. DEWALT D25553K 1-9/16-Inch Spline Combination Hammer Kit\n3. XtremepowerUS 10pc 1&quot; Dr.Deep Impact Cr-V Socket Set - MM (Black)\n4. Forney 60224 Mini Rotary File Cutter Set, 1/8-Inch Shaft, 3-Piece\n5. Voltec 08-00616 1400-Watt Halogen Pro Worklight, 7-Foot, Blue &amp; Yellow\n6. 3.5&quot; Acrylic Lens Rimless 2x Magnifying Glass w/2 LEDs - Great for Basic Inspections, Perfect for Crafts &amp; Hobbies!\n7. Icicle Solar Christmas String Lights, 15.7ft 8 Light Modes 20 LED Water Drop Fairy String Lighting for Outdoor &amp; Indoor, Home, Patio, Lawn, Garden, Party, and Holiday Decorations (Warm White)\n8. Blue Sea Systems 187 Series, 285 Series &amp; Klixon Circuit Breakers\n9. Ridgid 59832 Die Head Post\n10. Johnson Level &amp; Tool 175-L Post Level\n11. Wire Loom Black 20' Feet 1&quot; Split Tubing Hose Cover Auto Home Marine by Nippon America\n12. Brinks 7462-619 Hampton 3-Light Camille Bath Vanity Light, Satin Nickel\n13. Platform Stepladder, 7 ft. 9in, 330 lb.\n14. Self-Adhesive Stress Crack Tape Textured Roll\n15. SHURFLO (255-313) 1/2&quot; Twist-On Pipe Strainer\nYou should output 3 numbers that correspond to the selected products. There should be a comma separating every two numbers. Only respond with the results. Do not say any word or explanations.\nOutput: ":
            response = '8,11,15'
        elif prompt == "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive. \nInput: a review text for a(n) Leotard product\nRuns small and fabric doesn\u2019t stretch\nThis body shaper is way too small even though I bought 2X. It doesn\u2019t stretch at all to be form fitting. I don\u2019t like this product and will never purchase it again.\nOutput:\nAnswer: ":
            response = '1'
        elif prompt == "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive. \nInput: a review text for a(n) Boxing Glove product\nGood choice for begginers/intermediates\nGloves look exactly like on a picture, love them. Good gloves for good price, especially for beginners. I used to using leather gloves and that leather smell was really bad, I am glad that this time I decided to buy synthetic. Unfortunally this may be con too, because durabality is lower. Anyway I prefer this synthetic ones. Also part on palms is made with textile for ventilation, so hands don't sweat that much.\nOutput:\nAnswer: ":
            response = '4'
        elif prompt == "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive. \nInput: a review text for a(n) Thermometer product\nHas that cheaply made feel\nWhile I believe the thermometer to be accurate, it has the &#34;made in China&#34; cheaply made feel. Additionally, it is larger than I expected. I was hoping it would fit in one of the thermometer and pen pockets on the arm of my chef jacket, but it is too large and does not fit. I have a number of the Taylor thermometers that are much better quality.\nOutput:\nAnswer: ":
            response = '3'
        elif prompt == "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive. \nInput: a review text for a(n) optical frame product\nLove these!\nI really like these frames.  I like the style and the size of the lenses.  I would definitely buy them again in other colors.  Happy with this purchase.\nOutput:\nAnswer: ":
            response = '5'
        elif prompt == "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive. \nInput: a review text for a(n) Necklace product\nI bought this chain to go along with a alexander the great necklace and im gonna get right to the point if you want a 17 dollar necklace that looks like it came out of a gumball machine buy this one right here, its to short to shiny and to cheap to even consider wearing, go to mall try something nice on and buy it, stay away from this one.\nOutput:\nAnswer: ":
            response = '1'
        elif prompt == "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive. \nInput: a review text for a(n) Boxing Glove product\nPerfect for kids. I purchased them thinking they were toy boxing gloves but they seem pretty realy. I have a small hand and it doesn't fit but they are perfect size for my 6 and 7 yr olds.\nOutput:\nAnswer: ":
            response = '4'
        elif prompt == "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive. \nInput: a review text for a(n) Thermometer product\nThis unit works well, but the temp is approximately 4 degrees lower than actual temp. which is annoying.  The other two thermometers in my house were put next to this unit and they read consistent with each other but ~4 degrees hotter.  Not sure why the company can't make it so it reads a true temp.  However, the unit does work and fits my needs.\nOutput:\nAnswer: ":
            response = '3'
        elif prompt == "Instructions: Evaluate the following product review on a scale of 1 to 5, with 1 being very negative and 5 being very positive. \nInput: a review text for a(n) cereal product\nI love this Meusli. Its very simple. Im sure I could mix up my own, but its easier to just scoop it out for use.  I haven't tried cooking it. I usually put it in a bowl with a little almond milk and let it sit for about 10 minutes, which I think is what the package directions say. Id like to try cooking it sometime.  Its very good. Its currently my favorite cereal and, I think, much better for you than other packaged breakfast cereals.\nOutput:\nAnswer: ":
            response = '5'

        return response
