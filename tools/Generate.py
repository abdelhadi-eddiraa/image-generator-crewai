# import requests

# API_URL = "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
# headers = {"Authorization": "Bearer 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"}


# def query(payload):
# 	response = requests.post(API_URL, headers=headers, json=payload)
# 	return response.content
# image_bytes = query({
# 	"inputs": "Astronaut riding a horse",
# })
# # You can access the image with PIL.Image for example
# import io
# from PIL import Image
# image = Image.open(io.BytesIO(image_bytes))
# image.show()

import requests
import os
from PIL import Image
import io
from crewai_tools import BaseTool
from crewai_tools import tool
from typing import ClassVar

class GenerateTools():
   
    
    @tool
    def generate_image(prompt, output_path="generated_image.jpg"):
        """
        Args:
           prompt (str): the prompt string return from the classification agent is the argument prompt in the functiom generat image tool.
.
        """

    


        #init
        API_URL= "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers= {"Authorization": "Bearer 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"}
        # Combine the query logic within this function
        response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
        response.raise_for_status()  # Raise an exception for HTTP errors
        image_bytes = response.content
        
        # Process the image
        image = Image.open(io.BytesIO(image_bytes))
        image.save(output_path)
        print(f"Image generated and saved as {output_path}")
        image.show()
    

