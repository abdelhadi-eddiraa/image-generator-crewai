from textwrap import dedent
from crewai import Agent, Crew


from tasks import ImageClassificationTasks
from agents import ImageClassificationAgent







tasks = ImageClassificationTasks()
agents = ImageClassificationAgent()


print("## Welcome to the marketing Crew")
print('-------------------------------')





# Create Agents
image_classification_agent = agents.image_classification_agent()
image_generation_agent=agents.Generate_image_agent()

# Create Tasks
image_classification_task = tasks.analyze_product_images(image_classification_agent)
image_generation_task=tasks.generate_image_task(image_generation_agent)


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


Mycrew=classification_crew.kickoff()
# Print results
print("\n\n########################")
print("## Here is the result")
print("########################\n")
print("'\n\nYour midjourney description:")
print(Mycrew)
