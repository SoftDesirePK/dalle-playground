from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

import torch
import time

llm_chain = None

class FalconWrapper:
    def __init__(self) -> None:
        from transformers import AutoTokenizer
        from langchain import PromptTemplate, LLMChain

        model = "tiiuae/falcon-7b-instruct"

        tokenizer = AutoTokenizer.from_pretrained(model)

        import torch
        import transformers

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

        time.sleep(5)
        # Create the pipe schedular
        # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        # self.pipe = pipe.to("cuda")

        from langchain import HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=pipeline)

        template = """Question: {question}
        Answer: """
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

 

            
    def generate_query_response(self, text_prompt: str):

        question = text_prompt
        result = self.llm_chain.run(question)
        print(f"Warning: Result ->>>> {result}")

    
        return result
