from typing import List, Union
import random
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import transformers
import torch
import pandas as pd
import numpy as np
from .base_model import ShopBenchBaseModel
# import yake
import vllm
# import faiss
import sys
import json
# import faiss

#### CONFIG PARAMETERS ---

# Set a consistent seed for reproducibility
AICROWD_RUN_SEED = int(os.getenv("AICROWD_RUN_SEED", 773815))

# Batch size you wish the evaluators will use to call the `batch_generate_answer` function
AICROWD_SUBMISSION_BATCH_SIZE = 256 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
# AICROWD_SUBMISSION_BATCH_SIZE = 16 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.

# VLLM Parameters 
VLLM_TENSOR_PARALLEL_SIZE = 8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
# VLLM_TENSOR_PARALLEL_SIZE = 4 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.
VLLM_GPU_MEMORY_UTILIZATION = 0.8 # TUNE THIS VARIABLE depending on the number of GPUs you are requesting and the size of your model.



class Vicuna2ZeroShot(ShopBenchBaseModel):
    def __init__(self, model_path, instrution):
        random.seed(AICROWD_RUN_SEED)

        self.system_prompt = "You are a helpful online shopping assistant. Below is a example and a question. Please reference [Example] to  answer the following [Question] about online shopping and follow the given instructions. And you should answer in the [Answer] field.\n\n"
        self.query_item = pd.read_csv("./models/query_items.csv")
        self.RAG_model = SentenceTransformer("./models/snowflake-arctic-embed-m").cuda()
        # self.sentiment_tokenizer = AutoTokenizer.from_pretrained("/cbd-lizhipeng/Phi-3-mini-4k-instruct", trust_remote_code=True)
        # self.sentiment_model = AutoModelForCausalLM.from_pretrained("/cbd-lizhipeng/Phi-3-mini-4k-instruct", device_map='auto', torch_dtype=torch.float16, trust_remote_code=True, do_sample=True)

        self.instrution = instrution
        if "Qwen2".lower() in model_path.lower():
            gpu_cards = 4
        else:
            gpu_cards = 8
        self.llm = vllm.LLM(
            model_path,
            tensor_parallel_size=gpu_cards, 
            gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION, 
            trust_remote_code=True,
            dtype="float16", # note: bfloat16 is not supported on nvidia-T4 GPUs
            enforce_eager=True,
            # quantization="awq"
        )
        self.tokenizer = self.llm.get_tokenizer()

    def format_prommpts(self, prompts, is_multiple_choice, task_type):
        """
        Formats prompts using the chat_template of the model.
            
        Parameters:
        - queries (list of str): A list of queries to be formatted into prompts.
            
        """
        # system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping of the user and follow the given instructions. Please think cautiously. The answers or reponses are very important.\n\n"
        # if is_multiple_choice:
        #     system_prompt = ""
        # else:
        system_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping of the user and follow the given instructions.\n\n"

        formatted_prompts = []
        if self.instrution == 'self_cot_mcp':
            if task_type == 'multiple-choice':
                formatted_prompts = self.mc_cot(prompts)
            else:
                formatted_prompts.append([system_prompt + prompt for prompt in prompts])
        
        elif self.instrution == 'self_cot_all':
            formatted_prompts = self.mc_cot(prompt)
            
        elif self.instrution == 'little_model_cot_mcp':
            for prompt in prompts:
                if task_type == 'multiple-choice':
                    prompt = self.little_model_cot(prompt)
                    formatted_prompts.append(system_prompt + '\n\n' + prompt)
                else:
                    formatted_prompts.append(system_prompt + prompt)

        elif self.instrution == 'little_model_cot_all':
            for prompt in prompts:
                prompt = self.little_model_cot(prompt)
                formatted_prompts.append(prompt)

        elif self.instrution == 'generation_prompt':
            for prompt in prompts:
                if task_type == 'generation':
                    prompt = self.is_generate(prompt)
                formatted_prompts.append(prompt)
            else:
                formatted_prompts.append(system_prompt + prompt)

        elif self.instrution == 'similarity':
            for prompt in prompts:
                if task_type == 'ranking':
                    prompt = self.is_ranking(prompt)
                    formatted_prompts.append(prompt)
                else:
                    formatted_prompts.append(system_prompt + prompt)

        elif self.instrution == 'all':
            for prompt in prompts:
                if task_type == 'ranking':
                    prompt = self.is_ranking(prompt)
                
                if task_type == 'multiple-choice':
                    # prompt = self.mc_cot(prompts)
                    prompt = system_prompt + '\n\n' + prompt

                if task_type == 'generation':
                    prompt = self.is_generate(prompt)

                if task_type == "retrieval":
                    prompt = self.is_retrival(prompt)

                formatted_prompts.append(prompt)
        else:
            for prompt in prompts:
                formatted_prompts.append(system_prompt + '\n\n' + prompt)

        return formatted_prompts
    
    def generate_small(self, prompt: str, max_new_tokens: int, temperature: float) -> str:
        inputs = self.sentiment_tokenizer(prompt, return_tensors='pt')
        inputs.input_ids = inputs.input_ids.cuda()
        generate_ids = self.sentiment_model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, temperature=temperature)
        result = self.sentiment_tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        generation = result[len(prompt):]
        return generation
    
    def little_model_cot(self, prompt):
        prompt_info = prompt.replace('\nAnswer:','').replace('\nOutput:','')
        prompt_info = prompt_info.strip('\n') + '\nChain of thought step-bystep in 50 words: \n'

        ans = self.generate_small(prompt_info, 50, 0.5)

        prompt = '[Question]\n'+ prompt.split('Answer')[0] + '\n[Chain of Thought]\n' + ans.strip('\n') + '\n\n[Answer]\nTherefore, the answer is '

        return prompt

    def is_mc(self, prompt):
        # mc_prompt = "Suppose you are GPT-4. You can answer any questions asked by user. Now you are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions. Please think cautiously. The answers or reponses are very important. You must be sure your answer is right.\n\n"
        mc_prompt = "You're a product manager at Amazon. Please answer the following question about online shopping and follow the given instructions. Please think cautiously. The answers or reponses are very important.\n\n"
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
        return prompt
    
    def is_ranking(self, prompt):
        # if 'Query: ' not in prompt:
        #     return prompt
        # elif '????' in prompt:
        #     return prompt
        good_prompt = "Based on your previous knowledge then generate the answer. "
        ranking_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions.In this task, each question is associated with a requirement and a list of candidate items, and the model is required to re-rank all items according to how each item satisfies the requirement.\n\n"
        # ranking_prompt = "You are a helpful online shopping assistant. Please answer the following question about online shopping of the user and follow the given instructions.\n\n"
        query,item = self.split_query_item_task12(prompt)
        # related_item = self.use_kdd22_complete_query(query)
        similarity_out = self.process_similarity(query,item,5)
        # prompt = prompt.split('You should output ')[0]+'\n[External information] \n' + 'The following items are very important and have a strong relevance to the question, and ranked by relevance.\n' + '\n'.join(similarity_out) + '\nYou should output the randed list reference to the [External information]. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations. \nAnswer: '
        # prompt = prompt.split('You should output ')[0]+'\n[External information] \n'+related_item + '\nThe following items are very important and have a strong relevance to the query, and ranked by relevance.\n' + '\n'.join(similarity_out) + '\nYou should output a permutation of 1 to 5 reference to the order of [External information]. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations. \nOutput: '
        # prompt = prompt.split('You should output ')[0]+'\n[External information]\nThe following items are very important and have a strong relevance to the query, and ranked by relevance.\n' + '\n'.join(similarity_out) + '\nYou should output a permutation of 1 to 5 reference to the order of [External information]. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations. \nOutput: '
        prompt = prompt.split('You should output ')[0]+'\n[External information]\nThe following items are very important and have a strong relevance to the query, and ranked by relevance.\n' + '\n'.join(similarity_out) + '\nYou should output a permutation of 1 to 5. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations.\nAnswer (Only response oprion numbers and separated by a comma): '
        # prompt = prompt.split('You should output ')[0]+'\n[External information]' + '\nThe following items are very important and have a strong relevance to the query, and ranked by relevance.\n' + '\n'.join(similarity_out) + '\nYou should output the randed list reference to the [External information]. There should be a comma separating two numbers. Each product and its number should appear only once in the output. Only respond with the ranking results. Do not say any word or explanations. \nAnswer: '
        prompt = ranking_prompt + prompt
        return prompt
        
    def is_retrival(self, prompt):
        good_prompt = "Based on your previous knowledge then generate the answer. "
        retrival_prompt = "This is a retrieval question. You are a highly skilled online shopping assistant and a professional product retrieval expert. Your goal is to help consumers quickly and accurately identify products that meet their specific needs. You provide a clear and concise list of retrieval results, including the product name, key attributes, and how they meet the requirements. Please analyze the following request and deliver accurate retrieval results.\n\n"
        prompt = good_prompt + retrival_prompt + prompt

        return prompt

    def is_generate(self, prompt):
        good_prompt = "Based on your previous knowledge then generate the answer. "
        inst = self.select_inst(prompt)
        prompt = good_prompt + inst + prompt.replace('Answer','Response (limit to 30 words)').replace('Output','Response (limit to 30 words)')
        return prompt

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
        return inst
    
    def mc_cot(self, prompts):
        # mc_prompt2 = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions. Please think cautiously and deeply. The answers or reponses are very important. \n\n"
        # prompts = [mc_prompt2 + prompt for prompt in prompts]
        # return prompts
        prompts = [prompt.replace('\nAnswer:','').replace('\nOutput:','') for prompt in prompts]
        prompt_info = ['Provive your chain of thought step-by-step for this question: '+ prompt_info.strip('\n')  for prompt_info in prompts]

        ans = self.vllm_generate(prompt_info)
        mc_prompt2 = "You are a helpful online shopping assistant. Please answer the following question about online shopping of the user and follow the given instructions.\n\n"
        prompts_cot = []
        for a,p in zip(ans, prompts):
            if 'sentiment' not in p or 'similar' not in p:
            # if 'similar' not in p:
                prompts_cot.append(mc_prompt2 + p.strip()+ '\nAnswer: ')
            else:
                res = mc_prompt2 + '[Question]\n'+ p.strip() + '\n\n[Chain of Thought]\n' + a.strip('\n') + '\n\n[Answer]\nTherefore, the answer is '
                prompts_cot.append(res)

        return prompts_cot

    # def mc_cot(self, prompts):
    #     prompts = [prompt.replace('\nAnswer:','').replace('\nOutput:','') for prompt in prompts]
    #     prompt_info = [prompt_info.strip('\n') + '\nChain of thought step-bystep in 50 words: \n' for prompt_info in prompts]

    #     ans = self.vllm_generate(prompt_info)
    #     mc_prompt2 = "You are a helpful online shopping assistant. Please answer the following question about online shopping and follow the given instructions. Please think cautiously to be sure the right answer. The answers or reponses are very important.\n\n"
    #     prompts_cot = []
    #     for a,p in zip(ans, prompts):
    #         prompts_cot.append(mc_prompt2 + '[Question]\n' + p.split('Answer')[0] + '\n[Chain of Thought]\n' + a.strip('\n') + '\n\n[Answer]\nTherefore, the answer is ')

    #     return prompts_cot

    def vllm_generate(self, prompts):

        responses = self.llm.generate(
            prompts,
            vllm.SamplingParams(
                n=1,  # Number of output sequences to return for each prompt.
                top_p=0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                temperature=0,  # randomness of the sampling
                seed=AICROWD_RUN_SEED, # Seed for reprodicibility
                skip_special_tokens=True,  # Whether to skip special tokens in the output.
                max_tokens=150,  # Maximum number of tokens to generate per output sequence.
            ),
            use_tqdm = False
        )

        batch_response = []
        for response in responses:
            batch_response.append(response.outputs[0].text)        

        return batch_response

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

    def batch_predict(self, batch, is_multiple_choice:bool, task_type) -> List[str]:
        prompts = batch["prompt"]
        formatted_prompts = self.format_prommpts(prompts,is_multiple_choice, task_type)
        if is_multiple_choice:
            max_new_tokens = 2
        else:
            max_new_tokens = 70

        responses = self.llm.generate(
            formatted_prompts,
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

        return batch_response, formatted_prompts

    def cal_overlap_of_querise(self, query1,query2):
        q1_ls = query1.split(' ')
        q2_ls = query2.split(' ')
        overlap = 0
        for item in q1_ls:
            if item in q2_ls:
                overlap+=1
        res_txt = 'The second query ' + '\''+query2 + '\' contains {} words of query '.format(str(overlap)) +'\''+query1 + '\''
        return res_txt

    def find_items_of_query(self, query):
        # data_kdd22
        # query_item = pd.read_csv("query_items.csv") #写进self init里，防止每次打开大数据占时间
        # query_item.set_index(["query"], inplace=True)
        product_ls = self.query_item[self.query_item['query'] == query].values.tolist()

        related_products = self.query_item[self.query_item['query'] == query]['related_products'].values.tolist()
        print(product_ls)
        if len(related_products)>0:
            related_products = product_ls[0][1]
            unrelated_products = product_ls[0][2]
            # complement_products = product_ls[0][3]
            # substitute_products = product_ls[0][4]
        else:
            related_products="[]" 
            unrelated_products ="[]"
            # complement_products ="[]"
            # substitute_products ="[]"
        
        related_products = eval(related_products)
        unrelated_products = eval(unrelated_products)
        # complement_products = eval(complement_products)
        # substitute_products = eval(substitute_products)
        return related_products,unrelated_products#,complement_products,substitute_products

    def split_query_item_task12(self, text):
        try:
            if 'Query: ' not in text:
                text_ls = text.split('\'')
                query = text_ls[1]
                query = query.strip('\n').strip(' ')
            else:
                query = text.split('Query:')[1].split('\n')[0].strip()
                # query = query.strip('\n').strip(' ')
        except:
            query = text_ls = text.split('\n')[0]
        if 'Candidate product list:' in text:
            item_ls = text.split('Candidate product list:')[1].split('You should ')[0].strip('\n').strip(' ').strip('\n').split('\n')
        elif 'Product List:' in text:
            item_ls = text.split('Product List:')[1].split('You should ')[0].strip('\n').strip(' ').strip('\n').split('\n')
        else:
            item_ls = []
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
    
    # def get_keywords(self, text_ls):
    #     language = "en" # 文档语言
    #     max_ngram_size = 3 # N-grams
    #     deduplication_thresold = 0.2 # 筛选阈值,越小关键词越少
    #     deduplication_algo = 'seqm'
    #     windowSize = 1
    #     numOfKeywords = 5 # 最大数量
    #     kw_extractor = yake.KeywordExtractor(lan=language, 
    #                                     n=max_ngram_size, 
    #                                     dedupLim=deduplication_thresold, 
    #                                     dedupFunc=deduplication_algo, 
    #                                     windowsSize=windowSize, 
    #                                     top=numOfKeywords)
        
    #     item_keyword_ls = []
    #     for item in text_ls:
    #         text_res = item+' has keywords: '
    #         keywords = kw_extractor.extract_keywords(item)
    #         kw_ls = []
    #         for kw in keywords:
    #             kw_ls.append(kw[0])
    #         text_res=text_res+'; '.join(kw_ls)
    #         item_keyword_ls.append(text_res)
    #     return item_keyword_ls

    def process_similarity(self, query, item, k):
        RAG_model = self.RAG_model
        query_embeddings = RAG_model.encode(query, prompt_name="query")
        item_embeddings = RAG_model.encode(item)
        scores = query_embeddings @ item_embeddings.T
        doc_score_pairs = list(zip(item, scores))
        doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
        res = []
        k = min(k, len(doc_score_pairs))
        for i in range(k):
            res.append(doc_score_pairs[i][0])
        return res
    

    def split_query_item_task11(self, text):
        text_ls = text.split('\"')
        q1,q2 = text_ls[1],text_ls[3]
        query = q1.strip('\n').strip(' ').strip('\n')
        item = q2.strip('\n').strip(' ').strip('\n')

        return query,item

    def use_kdd22_complete_query_task14(self,query):
        # data_kdd22
        query_item = self.query_item #写进self init里，防止每次打开大数据占时间
        # query_item.set_index(["query"], inplace=True)
        related_products = query_item[query_item['query'] == query]['related_products'].values.tolist()
        if len(related_products)>0:
            related_products = related_products[0]
        else:
            related_products="[]"
        unrelated_products = query_item[query_item['query'] == query]['unrelated_products'].values.tolist()
        if len(unrelated_products)>0:
            unrelated_products = unrelated_products[0]
        else:
            unrelated_products="[]"
        related_products = eval(related_products)
        unrelated_products = eval(unrelated_products)
        for i in range(len(related_products)):
            related_products[i] = str(i+1)+'.'+related_products[i]
        for i in range(len(unrelated_products)):
            unrelated_products[i] = str(i+1)+'.'+unrelated_products[i]
        query_related_text = 'Here are some products which is exactly the query {} want: '.format(query)+ ';'.join(related_products)
        query_unrelated_text = 'Here are some products which is exactly the query {} do not want: '.format(query)+ ';'.join(unrelated_products)
        res_text =  query_related_text
        return res_text
    
    def rag_overall(self, query, k):
        RAG_model = self.RAG_model
        if "?\n" in query:
            query = query.split('?\n')[0]
        query_embeddings = RAG_model.encode(query, prompt_name="query")
        query_embeddings = np.expand_dims(query_embeddings, axis = 0)
        distance, query_rag_index = self.rag_index.search(query_embeddings, k)
        # print(query_rag_index)
        # print("---------------------------")
        res = []
        for i in range(k):
            res.append(self.rag_document[query_rag_index[0][i]])
        return res
    
    def optimize_rag(self,s):
        rag_opt = ''
        for i in s.split('|'):
            if 'Title:' in i:
                rag_opt = ('Description of the product:' + i.split(':')[1])
            elif 'Price:' in i:
                rag_opt += (', and its price is' + i.split(':')[1])

            elif 'Brand:' in i:
                rag_opt +=  (', and its brand is' + i.split(':')[1])

            elif 'Size:' in i:
                rag_opt += (', and its size is' + i.split(':')[1])
        return rag_opt
    
    # def RAG_inst_file(self, query):
    #     RAG_model = self.RAG_model
    #     query_embeddings = RAG_model.encode(query, prompt_name="query")
    #     query_embeddings = np.expand_dims(query_embeddings, axis = 0)
    #     D, I = self.inst_index.search(query_embeddings, 1)
    #     df_idx = I[0].tolist()
    #     res = self.inst_text.iloc[df_idx].values.tolist()
    #     res_all = []
    #     res_text = ''
    #     for item in res:
    #         # print(item)
    #         res_all.append(item[0])
    #         res_text = res_text +'*. '+ item[0] + '\n'
    #     return res_text
    
