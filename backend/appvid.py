import argparse
import base64
import os
from pathlib import Path
from io import BytesIO
import time

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
from stable_diffusion_wrapper import StableDiffusionWrapper
from consts import DEFAULT_IMG_OUTPUT_DIR, MAX_FILE_NAME_LEN
from utils import parse_arg_boolean

import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video

app = Flask(__name__)    #  Flask application object. 
CORS(app)        # allows the application to be accessed from other domains
print("--> Starting the image generation server. This might take up to two minutes.")

stable_diff_model = None

parser = argparse.ArgumentParser(description = "A text-to-image app to turn your textual prompts into visionary delights")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
parser.add_argument("--save_to_disk", type = parse_arg_boolean, default = False, help = "Should save generated images to disk")
parser.add_argument("--img_format", type = str.lower, default = "JPEG", help = "Generated images format", choices=['jpeg', 'png'])
parser.add_argument("--output_dir", type = str, default = DEFAULT_IMG_OUTPUT_DIR, help = "Customer directory for generated images")
args = parser.parse_args()

# GENERATE IMAGE
@app.route("/generate", methods=["POST"])
@cross_origin()
def generate_images_api():        # main endpoint for the application
    json_data = request.get_json(force=True)
    text_prompt = json_data["text"]
    num_images = json_data["num_images"]
    generated_imgs = stable_diff_model.generate_images(text_prompt, num_images)

    returned_generated_images = []
    if args.save_to_disk:
        dir_name = os.path.join(args.output_dir,f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{text_prompt}")[:MAX_FILE_NAME_LEN]
        Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    for idx, img in enumerate(generated_imgs):
        if args.save_to_disk: 
          img.save(os.path.join(dir_name, f'{idx}.{args.img_format}'), format=args.img_format)

        buffered = BytesIO()
        img.save(buffered, format=args.img_format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        returned_generated_images.append(img_str)

    print(f"Created {num_images} images from text prompt [{text_prompt}]")
    
    response = {'generatedImgs': returned_generated_images,
    'generatedImgsFormat': args.img_format}
    return jsonify(response)

# GENERATE VIDEO FRAMES
@app.route("/generatevdo", methods=["POST"])
@cross_origin()
def generate_frames_from_text_prompt():
    """Generates a sequence of frames from a text prompt.

    Args:
        json_data (dict): The JSON data containing the text prompt and the number of frames to generate.

    Returns:
        list: A list of the generated frames, encoded as base64 strings.
    """
    print("Generating Video")
    json_data = request.get_json(force=True)
    text_prompt = json_data["text"]
    num_frames = json_data["num_frames"]
    print(f"Parameter Received:\n json_data = {json_data}\n text_prompt = {text+prompt}\n num_frames = {num_frames}")
    pipe = DiffusionPipeline.from_pretrained("cerspense/zeroscope_v2_576w", torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    video_frames = pipe(text_prompt, num_inference_steps=20, height=320, width=576, num_frames=num_frames).frames
    # Export the sequence of frames to a video file.
    print("Exporting frames to video")
    video_path = export_to_video(video_frames) 

    # Return the video path as a response.
    response = jsonify({"video_path": video_path})
    return response



@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
    stable_diff_model = StableDiffusionWrapper()
    print("--> Image generation server is up and running!")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=False)
