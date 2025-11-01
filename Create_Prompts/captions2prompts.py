import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import itertools
from datasets import load_dataset, Dataset, DatasetDict
from PIL import Image
from huggingface_hub import login
from tqdm import tqdm


hf_token = "<HF_TOKEN>" # replace with your HuggingFace token
login(hf_token)

# Use the dataset generated with images2captions.py
cityscapes_captioned_hf_ds_id = "<DATASET_ID>"  # replace with your dataset ID on HuggingFace
dataset = load_dataset(cityscapes_captioned_hf_ds_id, split='train')

model_path = "meta-llama/Llama-3.1-8B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_path)
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)
 
generation_args = {
    "max_length": 512,
    "return_full_text": False,
    "do_sample": False,
}


REMOVE_INSTRUCTIONS = "camera is inside the car; logo; photo was taken inside the car; windshield; Mercedes."
REFINER_SYSTEM_CONTENT = ("You are an automated text refiner. "
                          "Refine the 'Refinable text' by removing/refining those statements, informations that are similar to 'Removable keywords'. "
                          "Also, the refined text should be compatible with 'Extra instructions' "
                          "Only give back the refined text, which will be used as a prompt. "
                         )

EXTRA_INSTRUCTIONS = [
    "It is raining.",
    "It is snowing.",
    "It is sunny.",
    "The weather is great, the sun is shining.",
    "The weather condition is overcast.",
    "The weather condition is foggy.",
    "It is early evening.",
    "It is sunset.",
    "The day is dawn/dusk.",
    "The weather is cloudy.",
    "It is a small town. The buildings are small.",
    "It is a poor part of a village.",
    "It is on a highway.", 
    "It is in India.",
    "It is in India. The buildings are small, old.",
    "The location is in a town in India.",
    "It is in India. It is raining.",
    "It is in India. The day is dawn/dusk.",
    "The camera is inside the car.",
    "It is early evening. The location is a city in India.",
    "The image is in a city",
    "The image is in a city",
    "The image is in a city",
]
EXTRA_INSTRUCTIONS_CYCLED = itertools.cycle(EXTRA_INSTRUCTIONS)

def generate_prompt(example):
    caption = example["caption"]
    extra_instruction = next(EXTRA_INSTRUCTIONS_CYCLED)
    caption = f"Removable keywords: {REMOVE_INSTRUCTIONS}, \n Extra instructions: {extra_instruction},\n Refinable text: {caption}"
    refiner_messages = [
        {"role": "system", "content": REFINER_SYSTEM_CONTENT},
        {"role": "user", "content": caption},
    ]
    #print(refiner_messages)
    output = pipe(refiner_messages, **generation_args)
    refined_caption = output[0]['generated_text']
    refined_caption = extra_instruction + " " + refined_caption
    return refined_caption

dataset = dataset.remove_columns(["image", "label"])
processed_examples = {"prompt": []}
for example in tqdm(dataset, desc="Processing"):
    processed_examples["prompt"].append(generate_prompt(example))

final_dataset = Dataset.from_dict(processed_examples)

# Push to hub
final_dataset.push_to_hub("flux_prompts", private=True)