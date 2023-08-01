import argparse
import base64
import os
from pathlib import Path
from io import BytesIO
import time

from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin

from falcon_wrapper import FalconWrapper
from consts import DEFAULT_IMG_OUTPUT_DIR, MAX_FILE_NAME_LEN
from utils import parse_arg_boolean

import numpy as np
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.utils import export_to_video
from flask import Response
# App

app = Flask(__name__)    #  Flask application object. 
CORS(app)        # allows the application to be accessed from other domains
print("--> Starting the flacon7b query server. This might take few minutes depending upon download speed or size of model.")


parser = argparse.ArgumentParser(description = "A text-to-image app to turn your textual prompts into visionary delights")
parser.add_argument("--port", type=int, default=8000, help = "backend port")
parser.add_argument("--save_to_disk", type = parse_arg_boolean, default = False, help = "Should save generated images to disk")
parser.add_argument("--img_format", type = str.lower, default = "JPEG", help = "Generated images format", choices=['jpeg', 'png'])
parser.add_argument("--output_dir", type = str, default = DEFAULT_IMG_OUTPUT_DIR, help = "Customer directory for generated images")
args = parser.parse_args()



llm_chain = None


# GENERATE FALCON7B RESPONSE - Last Modifed on 01-08-2023
@app.route("/queryfalcon", methods=["POST"])
@cross_origin()
def generate_response():
  
    print("Generating falcon response")
    json_data = request.get_json(force=True)
    text_prompt = json_data["text"]
    print(f"Parameter Received:\n json_data = {json_data}")
       
    response = ""
    return response
    


@app.route("/", methods=["GET"])
@cross_origin()
def health_check():
    return jsonify(success=True)


with app.app_context():
    # stable_diff_model = StableDiffusionWrapper()

    llm_chain = FalconWrapper()


    print("--> Falcon7b-instruct query server is up and running!")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=args.port, debug=True)
