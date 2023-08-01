from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import torch

class FalconWrapper:
    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from langchain import PromptTemplate, LLMChain
        import torch
        import transformers
        
        model = "tiiuae/falcon-7b-instruct"
        tokenizer = AutoTokenizer.from_pretrained(model)

        pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
        max_length=200,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
        )

        # Create the pipe schedular
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        self.pipe = pipe.to("cuda")

        # from langchain import HuggingFacePipeline
        # llm = HuggingFacePipeline(pipeline=pipeline)

        # question = "What is the capital of Saudi Arabia."
        # template = """Question: {question}
        # Answer: """
        # prompt = PromptTemplate(template=template, input_variables=["question"])
        # llm_chain = LLMChain(prompt=prompt, llm=llm)
        # result = llm_chain.run(question)
        # print(f"Warning: Test Result ->>>> {result}")

 

            
    def generate_query_response(self, text_prompt: str):
        from langchain import HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipeline)

        question = text_prompt
        template = """Question: {question}
        Answer: """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        result = llm_chain.run(question)
        print(f"Warning: Result ->>>> {result}")

    
        return result
