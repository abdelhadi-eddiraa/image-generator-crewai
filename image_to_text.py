
# import os
# from langchain_community.llms import HuggingFaceHub
# from crewai import Task, Crew, Agent
# from PIL import Image
# import requests
# from io import BytesIO





# # load image from the IAM database (actually this model is meant to be used on printed text)
# url = "https://raw.githubusercontent.com/ddobokki/ocr_img_example/master/g.jpg"
# response = requests.get(url)
# img = Image.open(BytesIO(response.content))
# preprocessed_image = img.resize((224, 224)) 
# print(preprocessed_image)

# # print(image)


# # Create a Language Model Manager (LLM) instance
# llm = HuggingFaceHub(
#     repo_id="ddobokki/ko-trocr",
#     huggingfacehub_api_token=os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
#     task="text-generation",
# )

# image_to_text_agent = Agent(
#     role="HuggingFace Agent",
#     goal="Generate text from a image using hangingdace",
#     backstory="A diligent explorer of image to text",
#     llm=llm
# )

# # Define a task
# image_to_text_task = Task(
#     description="Generate text from this image:{preprocessed_image} using the blip-image-captioning-base model.",
#     agent=image_to_text_agent,
#     expected_output="""A JSON OUTPUT like this example "generated_text": "two cats sleeping on a couch""",
# )

# # Define a crew with the language model manager as the agent and the text generation task
# text_generation_crew = Crew(
#     agents=[image_to_text_agent],
#     tasks=[image_to_text_task],
# )

# # Kick off the crew to execute the task
# crew_output = text_generation_crew.kickoff()

# # Print crew output
# print("Crew Output:")
# print(crew_output)
# from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
# import requests 
# import unicodedata
# from io import BytesIO
# from PIL import Image

# processor = TrOCRProcessor.from_pretrained("ddobokki/ko-trocr") 
# model = VisionEncoderDecoderModel.from_pretrained("ddobokki/ko-trocr")
# tokenizer = AutoTokenizer.from_pretrained("ddobokki/ko-trocr")

# url = "https://static.vecteezy.com/system/resources/previews/039/906/438/non_2x/activist-movement-protesting-against-racism-and-fighting-for-equality-demonstrators-from-different-cultures-and-race-protest-on-street-for-equal-rights-black-lives-matter-protests-city-concept-photo.jpg"
# response = requests.get(url)
# img = Image.open(BytesIO(response.content))

# pixel_values = processor(img, return_tensors="pt").pixel_values 
# generated_ids = model.generate(pixel_values, max_length=64)
# generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
# generated_text = unicodedata.normalize("NFC", generated_text)
# print(generated_text)


from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes

# Get all logits and their corresponding labels
logits_list = logits.squeeze().tolist()
labels = [model.config.id2label[i] for i in range(len(logits_list))]

# Combine logits and labels into a list of tuples
logits_and_labels = list(zip(labels, logits_list))

# Sort the list by logits in descending order
logits_and_labels.sort(key=lambda x: x[1], reverse=True)

# Print all logits and their corresponding labels
for label, logit in logits_and_labels:
    print(f"Class: {label}, Logit: {logit}")

