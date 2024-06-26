from crewai import Task
from textwrap import dedent

class ImageClassificationTasks:
    def analyze_product_images(self, agent):
        return Task(
            description=dedent(f"""\
                  Generate a search prompt based on image classification data.

Your task is to generate a string prompt using the provided tools and create a detailed and relevant search prompt based on the classification data.

Utilize your expertise in image classification to deliver precise and insightful prompts.

It's currently 2024.
Once you genrat a one prompt stop i only one one prompt.

               


                """),
            agent=agent,
            expected_output="""A STRING PROMPT for  serch the image"""
        )
    
    def generate_image_task(self,agent):

        return Task(
            description="""Generate an image using the prompt generated by the image_classification_agent.
             Once you genrat a one Image stop i only one one Image.
            
            important:
                        -once you generate the image stop .
                        -generat just one image
            """,
            agent=agent,
            expected_output="Generated image output here.",
            
            
        )

