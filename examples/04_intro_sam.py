import dotenv
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

from definitions import DATA_DIR
from transformer_agents.utils import iterate_image_files

config = dotenv.dotenv_values(".env")

if __name__ == "__main__":
    for image_file in iterate_image_files(f"{DATA_DIR}/images/Franka-Kitchen"):
        # Load an image
        image = Image.open(image_file).convert("RGB")

        input_point = [[30, 160]]  # A Nx2 array of point prompts to the model. Each point is in (X,Y) in pixels.
        # Display the image
        print(f"Filename: {image_file}")
        fig, ax = plt.subplots()
        ax.imshow(image)
        # Plot the point
        ax.scatter(input_point[0][0], input_point[0][1], c="r", s=10)
        ax.axis("off")
        fig.show()
        input("Press enter to continue...")

        # parameters
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if "cuda" in device else torch.float32
        model_type = "vit_h"
        checkpoint = "data/checkpoints/sam_vit_h_4b8939.pth"

        # SAM initialization
        model = sam_model_registry[model_type](checkpoint=checkpoint)
        model.to(device)
        predictor = SamPredictor(model)
        mask_generator = SamAutomaticMaskGenerator(model)
        predictor.set_image(np.array(image))  # load the image to predictor

        # Predict segmentation mask
        input_label = [
            1
        ]  # A length N array of labels for the point prompts. 1 indicates a foreground point and 0 indicates a background point.
        input_point = np.array(input_point)
        input_label = np.array(input_label)
        masks, scores, logits = predictor.predict(point_coords=input_point, point_labels=input_label)
        masks = masks[0, ...]

        # Crop image
        crop_mode = (  # Optional['wo_bg', 'w_bg'], where w_bg and wo_bg refer to remain and discard background separately.
            "wo_bg"
        )

        if crop_mode == "wo_bg":
            masked_image = image * masks[:, :, np.newaxis] + (1 - masks[:, :, np.newaxis]) * 255
            masked_image = np.uint8(masked_image)
        else:
            masked_image = np.array(image)
        masked_image = Image.fromarray(masked_image)

        from transformer_agents.utils import seg_to_box

        size = max(masks.shape[0], masks.shape[1])
        left, top, right, bottom = seg_to_box(
            masks, size
        )  # calculating the position of the top-left and bottom-right corners in the image
        print(left, top, right, bottom)

        image_crop = masked_image.crop((left * size, top * size, right * size, bottom * size))  # crop the image

        # Display the image
        print(f"Filename: {image_file}")
        fig, ax = plt.subplots()
        ax.imshow(masked_image)
        ax.axis("off")
        fig.show()

        inp = input("[Q]uit, or press any key to continue...\n")
        if inp.lower() == "q":
            break
