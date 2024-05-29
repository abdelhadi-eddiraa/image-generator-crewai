from PIL import Image
import requests
from langchain.tools import tool

from transformers import ViTImageProcessor, ViTForImageClassification

class PromptTools():
 
    @tool
    def classify_image(url):
        """ 
        Classify an image using the ViT model.
        Args:
           url (str): The URL of the image to classify.
       
        The `classify_image` tool requires a string like input
        Returns:
            list: A list of tuples containing logits and corresponding labels.
        """
        # url = 'https://static.vecteezy.com/system/resources/previews/039/906/438/non_2x/activist-movement-protesting-against-racism-and-fighting-for-equality-demonstrators-from-different-cultures-and-race-protest-on-street-for-equal-rights-black-lives-matter-protests-city-concept-photo.jpg'
        orgimage = Image.open(requests.get(url, stream=True).raw)

        
        # Resize the image to 244x244
        image = orgimage.resize((244, 244))

        # Initialize processor and model
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

        # Preprocess image and make predictions
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits

        # Get all logits and their corresponding labels
        logits_list = logits.squeeze().tolist()
        labels = [model.config.id2label[i] for i in range(len(logits_list))]

        # Combine logits and labels into a list of tuples
        logits_and_labels = list(zip(labels, logits_list))

        # Sort the list by logits in descending order
        logits_and_labels.sort(key=lambda x: x[1], reverse=True)

        # Return the sorted logits and their corresponding labels
        return logits_and_labels[:20]

# Example usage
# url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
# search_prompt = PromptTools.classify_image(url)
# print(search_prompt)
