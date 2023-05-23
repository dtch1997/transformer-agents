import pathlib

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

from definitions import DATA_DIR
from transformer_agents.utils import iterate_image_files

if __name__ == "__main__":
    # parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if "cuda" in device else torch.float32

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    captioning_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

    text = "Someone asked me if the microwave is closed or open. I said "

    for image_file in iterate_image_files(f"{DATA_DIR}/images/Franka-Kitchen"):
        # Load an image
        orig_fp = pathlib.Path(image_file).name
        orig_image = Image.open(image_file).convert("RGB")

        inputs = processor(orig_image, text, return_tensors="pt")
        out = captioning_model.generate(**inputs, max_new_tokens=50)
        captions = processor.decode(out[0], skip_special_tokens=True).strip()
        print(captions)

        # Load all masked images
        masked_fps = [pathlib.Path(image_file).stem + f"_masked_{i}.jpg" for i in range(3)]

        for masked_fp in masked_fps:
            masked_image = Image.open(masked_fp).convert("RGB")
            inputs = processor(masked_image, text, return_tensors="pt")
            out = captioning_model.generate(**inputs, max_new_tokens=50)
            captions = processor.decode(out[0], skip_special_tokens=True).strip()
            print(captions)

        break
