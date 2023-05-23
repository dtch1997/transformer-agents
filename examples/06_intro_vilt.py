import pathlib

import torch
from PIL import Image
from transformers import ViltForQuestionAnswering, ViltProcessor

from definitions import DATA_DIR
from transformer_agents.utils import iterate_image_files

if __name__ == "__main__":
    # parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    captioning_model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    text = "Is the microwave open or closed?"

    for image_file in iterate_image_files(f"{DATA_DIR}/images/Franka-Kitchen"):
        # Load an image
        orig_fp = pathlib.Path(image_file).name
        orig_image = Image.open(image_file).convert("RGB")

        encoding = processor(orig_image, text, return_tensors="pt")
        outputs = captioning_model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        print("Predicted answer:", captioning_model.config.id2label[idx])

        # Load all masked images
        masked_fps = [pathlib.Path(image_file).stem + f"_masked_{i}.jpg" for i in range(3)]

        for masked_fp in masked_fps:
            masked_image = Image.open(masked_fp).convert("RGB")
            encoding = processor(masked_image, text, return_tensors="pt")
            outputs = captioning_model(**encoding)
            logits = outputs.logits
            idx = logits.argmax(-1).item()
            print("Predicted answer:", captioning_model.config.id2label[idx])

        break
