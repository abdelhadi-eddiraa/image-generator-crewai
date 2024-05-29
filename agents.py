import os
from textwrap import dedent
from crewai import Agent
from langchain_groq import ChatGroq
from langchain_community.llms import  HuggingFaceEndpoint

from tools.Prompt import PromptTools
from tools.Generate import GenerateTools

class ImageClassificationAgent:
    def __init__(self):
        self.groqllm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="llama3-8b-8192"
        )
       
        

    def image_classification_agent(self,url):
        """\
 
    Defines the Image Classification Agent properties.

    Args:
      url (str): The URL of the image for classification.

    Returns:
      str: The generated prompt based on the image classification.
    
    """
        return Agent(
            role="Image Generate Prompt Agent",
#             goal=dedent("""\
# Based on the data returned from the tools, generate a string prompt that can be used for searching images. The higher number associated with each label indicates that the corresponding object is more prominent in the image.
# pass the url:{url} to the tools. The function classify_image expects a url as string; 

#                         !important:
    

#     -pass the {url} to the to she expcted argument url
#     -Make sure to include all the keys in your prompt without forgetting any. Each key represents an object detected in the image, and the higher the associated number, the more prominent that object is in the image
#     -Utilize all the data from the tools to create a comprehensive search prompt..
#     -Emphasize the labels with higher numbers, indicating their prominence in the image..
#     -Your prompt should be descriptive and contain more than 15 words.

                        
# .

#                """),
goal=dedent("""\
Based on the data returned from the tools, generate a string prompt that can be used for searching images. The higher number associated with each label indicates that the corresponding object is more prominent in the image.
Pass the URL: {url} to the tool. The function classify_image expects a URL as a string.
Once you genrat a one prompt stop i only one one prompt.
once you generate the prompt pass it to the next agent Generate_image_agent.
!important:
  - Pass the {url} to the tool, as it expects an argument named 'url'.
  - Make sure to include all the keys in your prompt without forgetting any. Each key represents an object detected in the image, and the higher the associated number, the more prominent that object is in the image.
  - Utilize all the data from the tools to create a comprehensive search prompt.
  - Emphasize the labels with higher numbers, indicating their prominence in the image.
  - Your prompt should be descriptive and contain more than 15 words.

""".format(url=url)),  # Use string formatting to insert the url

            backstory=dedent("""\
                As an Image Generate Prompt Agent at a top-tier
                digital marketing firm, your expertise lies in analyzing
                image classification data and generating prompts to support
                marketing strategies with precise and actionable insights."""),
            tools=[PromptTools.classify_image],
            allow_delegation=True,
            llm=self.groqllm,
            verbose=True,
            expected_input={"url": url}

           
        )
    
    def Generate_image_agent(self):
        return Agent(
            role="Image Generate Image Agent",
            goal=dedent("""\
                Based on the prompt generated from the image_classification_agent.
                pass the prompt to the tools in the Generate_image_agent. The function generate_image expects a prompt. 
                pass the prompt from the agent image_classification_agent and generate an image.
                Once you genrat a one Image stop i only one one Image.
                important:
                        -once you generate the image stop .
                        -generat one image
                """),
            backstory=dedent("""\
                As an Image Generate Image Agent at a top-tier
                digital marketing firm, your expertise lies in generating
                high-quality images based on descriptive prompts.
                """),
            tools=[GenerateTools.generate_image],
            allow_delegation=True,
            llm=self.groqllm,
            verbose=True
        )

