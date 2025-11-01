from huggingface_hub import login
from datasets import load_dataset, Dataset
from diffusers import FluxPipeline, FluxTransformer2DModel, GGUFQuantizationConfig
import torch
from PIL import Image
from tqdm import tqdm
import os

hf_token = "<HF_TOKEN>" # replace with your HuggingFace token
login(hf_token)

NUM_INFERENCE_STEPS = 30
IMAGE_SIZE = [1024, 512]
SAVE_IMAGES_LOCALLY = True
OUTPUT_PATH = "generated_images" # only relevant if SAVE_IMAGES_LOCALLY is True

dataset = load_dataset('regpeter/flux_prompts', split='train')

ckpt_path = (
    "https://huggingface.co/city96/FLUX.1-dev-gguf/blob/main/flux1-dev-Q8_0.gguf"
)
transformer = FluxTransformer2DModel.from_single_file(
    ckpt_path,
    quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
    torch_dtype=torch.bfloat16,
)
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()
# generator = torch.Generator().manual_seed(4)

def generate_image(prompt, prompt2=None):    
    output = pipe(
        prompt,
        prompt2,
        width=IMAGE_SIZE[0],
        height=IMAGE_SIZE[1],
        guidance_scale=3.5,
        num_inference_steps=NUM_INFERENCE_STEPS,
        max_sequence_length=512,
        # generator=generator,
    )
    return output.images[0]

image_list = []
for i, example in enumerate(tqdm(dataset)):
    # if i < 2599:  # Skip the first N examples (they are already generated)
    #     continue
    caption = example['prompt']

    # ----------
    # NOTE: Use this prompt for normal road images
    # prompt = f"The camera is on the right side of the road. Right-handed traffic. Realistic, natural, high quality, photo-realistic. {caption}"
    # ----------
    
    # ----------
    # NOTE: Use this prompt for normal road images with intersection
    prompt = f"The image shows an intersection of two roads, a main road and a side road. The camera is on the side road."
    prompt2 = (
        f"The camera is on the right side of the road in an intersection. Right-handed traffic. "
        f"Realistic, natural, high quality, photo-realistic. The image shows a street corner "
        f"and the intersection with a perpendicular main road. The side road joins the main road in the intersection, "
        f"the main road runs horizontally on the image. {caption}"
    )
    # ----------

    # ----------
    # NOTE: Use this prompt for drawing style road images with intersection
    # prompt = f"Use drawing style. The image is a colorful drawing of another photo. Drawing styled image, not photo-realistic. The image shows an intersection of two roads, a main road and a side road. The camera is on the side road."
    # prompt2 = (
    #     f"Use drawing style. The image is a colorful drawing of another photo. The camera is on the right side of the road in an intersection. Right-handed traffic. "
    #     # f"Realistic, natural, high quality, photo-realistic. The image shows a street corner "
    #     f"Drawing styled image, not photo-realistic. The image shows a street corner "
    #     f"and the intersection with a perpendicular main road. The side road joins the main road in the intersection, "
    #     f"the main road runs horizontally on the image. {caption}"
    # )
    # ----------
    
    gen_image = generate_image(prompt, prompt2)
    if SAVE_IMAGES_LOCALLY:
        os.makedirs(OUTPUT_PATH, exist_ok=True)
        save_path = os.path.join(OUTPUT_PATH, f"{i+1:06d}.png")
        gen_image.save(save_path)

    image_list.append(gen_image)

new_dataset = Dataset.from_dict({
    'image': image_list
})

new_dataset.push_to_hub("Flux_generated_images", private=True)