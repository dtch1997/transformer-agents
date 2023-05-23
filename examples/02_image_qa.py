import dotenv
import matplotlib.pyplot as plt
from PIL import Image
from transformers.tools import load_tool

from definitions import DATA_DIR

config = dotenv.dotenv_values(".env")

if __name__ == "__main__":
    # Load an image
    img = Image.open(f"{DATA_DIR}/franka_kitchen_1.jpg")

    img_qa_tool = load_tool("image-question-answering")

    # Display the image
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.axis("off")
    fig.show()
    input("Press enter to continue...")

    # Print the answer
    questions = ["What is the color of the fridge?", "Is the microwave open?", "What is the robot doing?"]

    for question in questions:
        answer = img_qa_tool(img, question)
        print(f"Q: {question}\nA: {answer}\n")
