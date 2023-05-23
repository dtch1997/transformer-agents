import dotenv
import matplotlib.pyplot as plt
from transformers.tools import OpenAiAgent

config = dotenv.dotenv_values(".env")

if __name__ == "__main__":
    agent = OpenAiAgent(model="text-davinci-003", api_key=config["OPENAI_API_KEY"])
    print("OpenAI is initialized 💪")

    # Ask a question
    boat = agent.run("Generate an image of a boat in the water")

    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(boat)
    ax.axis("off")
    fig.show()
    input("Press enter to continue...")
