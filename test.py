

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


import os
from textwrap import dedent
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent, Crew
from tasks import ImageClassificationTasks
from agents import ImageClassificationAgent


from tools.Prompt import PromptTools
from tools.Generate import GenerateTools

# Initialize FastAPI app
app = FastAPI()

# Initialize tasks and agents
tasks = ImageClassificationTasks()
agents = ImageClassificationAgent()

# Agent and Task Initialization
def initialize_agents():
    image_classification_agent = agents.image_classification_agent()
    image_generation_agent = agents.Generate_image_agent()
    return image_classification_agent, image_generation_agent

def initialize_tasks(image_classification_agent, image_generation_agent):
    image_classification_task = tasks.analyze_product_images(image_classification_agent)
    image_generation_task = tasks.generate_image_task(image_generation_agent)
    return image_classification_task, image_generation_task

class ImageRequest(BaseModel):
    url: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the Image Processing API"}

@app.post("/process-image")
def process_image(request: ImageRequest):
    try:

        # initila the tool to expected the url argument
       

        # Create Agents
        image_classification_agent = agents.image_classification_agent(url=request.url)
        image_generation_agent = agents.Generate_image_agent()
        
        # Create Tasks
        image_classification_task = tasks.analyze_product_images(image_classification_agent)
        image_generation_task = tasks.generate_image_task(image_generation_agent)
        
        # Create Crew responsible for Copy
        classification_crew = Crew(
            agents=[
                image_classification_agent,
                image_generation_agent
            ],
            tasks=[
                image_classification_task,
                image_generation_task
            ],
            verbose=True
        )

        # Kickoff the crew
        crew_result = classification_crew.kickoff()

        # Return the result
        return {
            "result": crew_result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)







