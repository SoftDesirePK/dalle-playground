from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

class StableDiffusionWrapper:
    def __init__(self) -> None:
        repo_id = "stabilityai/stable-diffusion-2-base"
        repo_id = "stabilityai/stable-diffusion-2-1"
        #repo_id = "stabilityai/stable-diffusion-xl-base-0.9"

        if repo_id == "stabilityai/stable-diffusion-2-base":
            pipe = DiffusionPipeline.from_pretrained(
                repo_id, revision="fp16",
                torch_dtype=torch.float16
            )

        if repo_id == "stabilityai/stable-diffusion-2-1":
            pipe = StableDiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            pipe = pipe.to("cuda")
        
        if repo_id == "stabilityai/stable-diffusion-xl-base-0.9":
            pipe = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
            pipe.to("cuda")
            
        
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config)
        self.pipe = pipe.to("cuda")

            
    def generate_images(self, text_prompt: str, num_images: int):
        prompt = [text_prompt] * num_images
        images = self.pipe(prompt, num_inference_steps=10).images
        return images
