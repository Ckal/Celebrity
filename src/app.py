import os
import gradio as gr
import json
import logging
import torch
from PIL import Image
import spaces
from diffusers import DiffusionPipeline, AutoencoderTiny, AutoencoderKL, AutoPipelineForImage2Image
from live_preview_helpers import calculate_shift, retrieve_timesteps, flux_pipe_call_that_returns_an_iterable_of_images
from huggingface_hub import hf_hub_download, HfFileSystem, ModelCard, snapshot_download
from transformers import AutoModelForCausalLM, CLIPTokenizer, CLIPProcessor, CLIPModel, LongformerTokenizer, LongformerModel
import copy
import random
import time
import requests
import pandas as pd

# Disable tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the CLIP tokenizer and model
clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")

# Initialize the Longformer tokenizer and model
longformer_tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
longformer_model = LongformerModel.from_pretrained("allenai/longformer-base-4096")

#Load prompts for randomization
df = pd.read_csv('prompts.csv', header=None)
prompt_values = df.values.flatten()

# Load LoRAs from JSON file
with open('loras.json', 'r') as f:
    loras = json.load(f)

# Initialize the base model
dtype = torch.bfloat16
device = "cuda" if torch.cuda.is_available() else "cpu"
base_model = "sayakpaul/FLUX.1-merged"

taef1 = AutoencoderTiny.from_pretrained("madebyollin/taef1", torch_dtype=dtype).to(device)
good_vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae", torch_dtype=dtype).to(device)
pipe = DiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype, vae=taef1).to(device)
pipe_i2i = AutoPipelineForImage2Image.from_pretrained(
    base_model,
    vae=good_vae,
    transformer=pipe.transformer,
    text_encoder=pipe.text_encoder,
    tokenizer=pipe.tokenizer,
    text_encoder_2=pipe.text_encoder_2,
    tokenizer_2=pipe.tokenizer_2,
    torch_dtype=dtype
)
MAX_SEED = 2**32 - 1

pipe.flux_pipe_call_that_returns_an_iterable_of_images = flux_pipe_call_that_returns_an_iterable_of_images.__get__(pipe)

def process_input(input_text):
    # Tokenize and truncate input
    #inputs = clip_processor(text=input_text, return_tensors="pt", padding=True, truncation=True, max_length=77)
    #return inputs
    #Change clip_processor to longformer
    inputs = longformer_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=4096)
    return inputs

# Example usage
input_text = "Your long prompt goes here..."
inputs = process_input(input_text)

class calculateDuration:
    def __init__(self, activity_name=""):
        self.activity_name = activity_name

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        if self.activity_name:
            print(f"Elapsed time for {self.activity_name}: {self.elapsed_time:.6f} seconds")
        else:
            print(f"Elapsed time: {self.elapsed_time:.6f} seconds")

def download_file(url, directory=None):
    if directory is None:
        directory = os.getcwd()  # Use current working directory if not specified
    
    # Get the filename from the URL
    filename = url.split('/')[-1]
    
    # Full path for the downloaded file
    filepath = os.path.join(directory, filename)
    
    # Download the file
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for bad status codes
    
    # Write the content to the file
    with open(filepath, 'wb') as file:
        file.write(response.content)
    
    return filepath

def update_selection(evt: gr.SelectData, selected_indices, loras_state, width, height):
    selected_index = evt.index
    selected_indices = selected_indices or []
    if selected_index in selected_indices:
        selected_indices.remove(selected_index)
    else:
        if len(selected_indices) < 4:
            selected_indices.append(selected_index)
        else:
            gr.Warning("You can select up to 4 LoRAs, remove one to select a new one.")
            return gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), selected_indices, gr.update(), gr.update(), gr.update(), gr.update(), width, height, gr.update(), gr.update(), gr.update(), gr.update()

    selected_info_1 = "Select a Celebrity as LoRA 1"
    selected_info_2 = "Select a LoRA 2"
    selected_info_3 = "Select a LoRA 3"
    selected_info_4 = "Select a LoRA 4"
    lora_scale_1 = 1.15
    lora_scale_2 = 1.15
    lora_scale_3 = 0.65
    lora_scale_4 = 0.65
    lora_image_1 = None
    lora_image_2 = None
    lora_image_3 = None
    lora_image_4 = None
    if len(selected_indices) >= 1:
        lora1 = loras_state[selected_indices[0]]
        selected_info_1 = f"### LoRA 1 Selected: [{lora1['title']}](https://huggingface.co/{lora1['repo']}) ✨"
        lora_image_1 = lora1['image']
        
    if len(selected_indices) >= 2:
        lora2 = loras_state[selected_indices[1]]
        selected_info_2 = f"### LoRA 2 Selected: [{lora2['title']}](https://huggingface.co/{lora2['repo']}) ✨"
        lora_image_2 = lora2['image']
        
    if len(selected_indices) >= 3:
        lora3 = loras_state[selected_indices[2]]
        selected_info_3 = f"### LoRA 3 Selected: [{lora3['title']}](https://huggingface.co/{lora3['repo']}) ✨"
        lora_image_3 = lora3['image']
        
    if len(selected_indices) >= 4:
        lora4 = loras_state[selected_indices[3]]
        selected_info_4 = f"### LoRA 4 Selected: [{lora4['title']}](https://huggingface.co/{lora4['repo']}) ✨"
        lora_image_4 = lora4['image']

    if selected_indices:
        last_selected_lora = loras_state[selected_indices[-1]]
        new_placeholder = f"Type a prompt for {last_selected_lora['title']}"
    else:
        new_placeholder = "Type a prompt after selecting a LoRA"

    return gr.update(placeholder=new_placeholder), selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, width, height, lora_image_1, lora_image_2, lora_image_3, lora_image_4

def remove_lora_1(selected_indices, loras_state):
    if len(selected_indices) >= 1:
        selected_indices.pop(0)
    selected_info_1 = "Select a Celebrity as LoRA 1"
    selected_info_2 = "Select a LoRA 2"
    selected_info_3 = "Select a LoRA 3"
    selected_info_4 = "Select a LoRA 4"
    lora_scale_1 = 1.15
    lora_scale_2 = 1.15
    lora_scale_3 = 0.65
    lora_scale_4 = 0.65
    lora_image_1 = None
    lora_image_2 = None
    lora_image_3 = None
    lora_image_4 = None
    if len(selected_indices) >= 1:
        lora1 = loras_state[selected_indices[0]]
        selected_info_1 = f"### LoRA 1 Selected: [{lora1['title']}](https://huggingface.co/{lora1['repo']}) ✨"
        lora_image_1 = lora1['image']
        
    if len(selected_indices) >= 2:
        lora2 = loras_state[selected_indices[1]]
        selected_info_2 = f"### LoRA 2 Selected: [{lora2['title']}](https://huggingface.co/{lora2['repo']}) ✨"
        lora_image_2 = lora2['image']
        
    if len(selected_indices) >= 3:
        lora3 = loras_state[selected_indices[2]]
        selected_info_3 = f"### LoRA 3 Selected: [{lora3['title']}](https://huggingface.co/{lora3['repo']}) ✨"
        lora_image_3 = lora3['image']
        
    if len(selected_indices) >= 4:
        lora4 = loras_state[selected_indices[3]]
        selected_info_4 = f"### LoRA 4 Selected: [{lora4['title']}](https://huggingface.co/{lora4['repo']}) ✨"
        lora_image_4 = lora4['image']
    return selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4

def remove_lora_2(selected_indices, loras_state):
    if len(selected_indices) >= 2:
        selected_indices.pop(1)
    selected_info_1 = "Select a Celebrity as LoRA 1"
    selected_info_2 = "Select a LoRA 2"
    selected_info_3 = "Select a LoRA 3"
    selected_info_4 = "Select a LoRA 4"
    lora_scale_1 = 1.15
    lora_scale_2 = 1.15
    lora_scale_3 = 0.65
    lora_scale_4 = 0.65
    lora_image_1 = None
    lora_image_2 = None
    lora_image_3 = None
    lora_image_4 = None
    if len(selected_indices) >= 1:
        lora1 = loras_state[selected_indices[0]]
        selected_info_1 = f"### LoRA 1 Selected: [{lora1['title']}](https://huggingface.co/{lora1['repo']}) ✨"
        lora_image_1 = lora1['image']
        
    if len(selected_indices) >= 2:
        lora2 = loras_state[selected_indices[1]]
        selected_info_2 = f"### LoRA 2 Selected: [{lora2['title']}](https://huggingface.co/{lora2['repo']}) ✨"
        lora_image_2 = lora2['image']
        
    if len(selected_indices) >= 3:
        lora3 = loras_state[selected_indices[2]]
        selected_info_3 = f"### LoRA 3 Selected: [{lora3['title']}](https://huggingface.co/{lora3['repo']}) ✨"
        lora_image_3 = lora3['image']
        
    if len(selected_indices) >= 4:
        lora4 = loras_state[selected_indices[3]]
        selected_info_4 = f"### LoRA 4 Selected: [{lora4['title']}](https://huggingface.co/{lora4['repo']}) ✨"
        lora_image_4 = lora4['image']
    return selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4

def remove_lora_3(selected_indices, loras_state):
    if len(selected_indices) >= 3:
        selected_indices.pop(2)
    selected_info_1 = "Select a Celebrity as LoRA 1"
    selected_info_2 = "Select a LoRA 2"
    selected_info_3 = "Select a LoRA 3"
    selected_info_4 = "Select a LoRA 4"
    lora_scale_1 = 1.15
    lora_scale_2 = 1.15
    lora_scale_3 = 0.65
    lora_scale_4 = 0.65
    lora_image_1 = None
    lora_image_2 = None
    lora_image_3 = None
    lora_image_4 = None
    if len(selected_indices) >= 1:
        lora1 = loras_state[selected_indices[0]]
        selected_info_1 = f"### LoRA 1 Selected: [{lora1['title']}](https://huggingface.co/{lora1['repo']}) ✨"
        lora_image_1 = lora1['image']
        
    if len(selected_indices) >= 2:
        lora2 = loras_state[selected_indices[1]]
        selected_info_2 = f"### LoRA 2 Selected: [{lora2['title']}](https://huggingface.co/{lora2['repo']}) ✨"
        lora_image_2 = lora2['image']
        
    if len(selected_indices) >= 3:
        lora3 = loras_state[selected_indices[2]]
        selected_info_3 = f"### LoRA 3 Selected: [{lora3['title']}](https://huggingface.co/{lora3['repo']}) ✨"
        lora_image_3 = lora3['image']
        
    if len(selected_indices) >= 4:
        lora4 = loras_state[selected_indices[3]]
        selected_info_4 = f"### LoRA 4 Selected: [{lora4['title']}](https://huggingface.co/{lora4['repo']}) ✨"
        lora_image_4 = lora4['image']
    return selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4

def remove_lora_4(selected_indices, loras_state):
    if len(selected_indices) >= 4:
        selected_indices.pop(3)
    selected_info_1 = "Select a Celebrity as LoRA 1"
    selected_info_2 = "Select a LoRA 2"
    selected_info_3 = "Select a LoRA 3"
    selected_info_4 = "Select a LoRA 4"
    lora_scale_1 = 1.15
    lora_scale_2 = 1.15
    lora_scale_3 = 0.65
    lora_scale_4 = 0.65
    lora_image_1 = None
    lora_image_2 = None
    lora_image_3 = None
    lora_image_4 = None
    if len(selected_indices) >= 1:
        lora1 = loras_state[selected_indices[0]]
        selected_info_1 = f"### Celebrity Selected: [{lora1['title']}](https://huggingface.co/{lora1['repo']}) ✨"
        lora_image_1 = lora1['image']
        
    if len(selected_indices) >= 2:
        lora2 = loras_state[selected_indices[1]]
        selected_info_2 = f"### LoRA 2 Selected: [{lora2['title']}](https://huggingface.co/{lora2['repo']}) ✨"
        lora_image_2 = lora2['image']
        
    if len(selected_indices) >= 3:
        lora3 = loras_state[selected_indices[2]]
        selected_info_3 = f"### LoRA 3 Selected: [{lora3['title']}](https://huggingface.co/{lora3['repo']}) ✨"
        lora_image_3 = lora3['image']
        
    if len(selected_indices) >= 4:
        lora4 = loras_state[selected_indices[3]]
        selected_info_4 = f"### LoRA 4 Selected: [{lora4['title']}](https://huggingface.co/{lora4['repo']}) ✨"
        lora_image_4 = lora4['image']
    return selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4
    
def randomize_loras(selected_indices, loras_state):
    if len(loras_state) < 2:
        raise gr.Error("Not enough LoRAs to randomize.")
    selected_indices = random.sample(range(len(loras_state)), 2)
    lora1 = loras_state[selected_indices[0]]
    lora2 = loras_state[selected_indices[1]]
    selected_info_1 = f"### LoRA 1 Selected: [{lora1['title']}](https://huggingface.co/{lora1['repo']}) ✨"
    selected_info_2 = f"### LoRA 2 Selected: [{lora2['title']}](https://huggingface.co/{lora2['repo']}) ✨"
    lora_scale_1 = 1.15
    lora_scale_2 = 1.15
    lora_image_1 = lora1['image']
    lora_image_2 = lora2['image']
    random_prompt = random.choice(prompt_values)
    return selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4, random_prompt

def add_custom_lora(custom_lora, selected_indices, current_loras, gallery, request: gr.Request = None):
    if not custom_lora:
        return current_loras, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), selected_indices, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
    
    try:
        # Retrieve user token if running in Spaces
        user_token = request.headers.get("Authorization", "").replace("Bearer ", "") if request else None

        # Check and load custom LoRA
        title, repo, path, trigger_word, image = check_custom_model(custom_lora, token=user_token)
        print(f"Loaded custom LoRA: {repo}")
        
        # Check if the LoRA already exists in the current list
        existing_item_index = next((index for (index, item) in enumerate(current_loras) if item['repo'] == repo), None)
        
        if existing_item_index is None:
            # Download if a direct .safetensors URL
            if repo.endswith(".safetensors") and repo.startswith("http"):
                repo = download_file(repo)
            
            # Add the new LoRA
            new_item = {
                "image": image or "/home/user/app/custom.png",
                "title": title,
                "repo": repo,
                "weights": path,
                "trigger_word": trigger_word,
            }
            print(f"New LoRA: {new_item}")
            existing_item_index = len(current_loras)
            current_loras.append(new_item)
        
        # Update gallery items
        gallery_items = [(item["image"], item["title"]) for item in current_loras]

        # Update selected indices
        if len(selected_indices) < 4:
            selected_indices.append(existing_item_index)
        else:
            raise gr.Error("You can select up to 4 LoRAs. Please remove one to add a new one.")

        # Update selection info and images
        selected_info = [f"Select a LoRA {i + 1}" for i in range(4)]
        lora_images = [None] * 4
        lora_scales = [1.15, 1.15, 0.65, 0.65]

        for idx, sel_idx in enumerate(selected_indices[:4]):
            lora = current_loras[sel_idx]
            selected_info[idx] = f"### LoRA {idx + 1} Selected: {lora['title']} ✨"
            lora_images[idx] = lora.get("image")

        print("Finished adding custom LoRA")
        return (
            current_loras,
            gr.update(value=gallery_items),
            *selected_info,
            selected_indices,
            *lora_scales,
            *lora_images,
        )

    except Exception as e:
        print(e)
        return (current_loras, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), selected_indices, gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(), gr.update(),gr.update(),
        )

def process_custom_lora(custom_lora, request: gr.Request):
    # Extract user token from request headers
    user_token = request.headers.get("Authorization", "").replace("Bearer ", "")
    if not user_token:
        raise gr.Error("User is not logged in. Please log in to use this feature.")
    return check_custom_model(custom_lora, token=user_token)

def remove_custom_lora(selected_indices, current_loras, gallery):
    if current_loras:
        custom_lora_repo = current_loras[-1]['repo']
        # Remove from loras list
        current_loras = current_loras[:-1]
        # Remove from selected_indices if selected
        custom_lora_index = len(current_loras)
        if custom_lora_index in selected_indices:
            selected_indices.remove(custom_lora_index)
    # Update gallery
    gallery_items = [(item["image"], item["title"]) for item in current_loras]
    # Update selected_info and images
    selected_info_1 = "Select a Celebrity as LoRA 1"
    selected_info_2 = "Select a LoRA 2"
    selected_info_3 = "Select a LoRA 3"
    selected_info_4 = "Select a LoRA 4"
    lora_scale_1 = 1.15
    lora_scale_2 = 1.15
    lora_scale_3 = 0.65
    lora_scale_4 = 0.65
    lora_image_1 = None
    lora_image_2 = None
    lora_image_3 = None
    lora_image_4 = None
    if len(selected_indices) >= 1:
        lora1 = loras_state[selected_indices[0]]
        selected_info_1 = f"### LoRA 1 Selected: [{lora1['title']}](https://huggingface.co/{lora1['repo']}) ✨"
        lora_image_1 = lora1['image']
        
    if len(selected_indices) >= 2:
        lora2 = loras_state[selected_indices[1]]
        selected_info_2 = f"### LoRA 2 Selected: [{lora2['title']}](https://huggingface.co/{lora2['repo']}) ✨"
        lora_image_2 = lora2['image']
        
    if len(selected_indices) >= 3:
        lora3 = loras_state[selected_indices[2]]
        selected_info_3 = f"### LoRA 3 Selected: [{lora3['title']}](https://huggingface.co/{lora3['repo']}) ✨"
        lora_image_3 = lora3['image']
        
    if len(selected_indices) >= 4:
        lora4 = loras_state[selected_indices[3]]
        selected_info_4 = f"### LoRA 4 Selected: [{lora4['title']}](https://huggingface.co/{lora4['repo']}) ✨"
        lora_image_4 = lora4['image']
    return (current_loras, gr.update(value=gallery_items), selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4)

def generate_image(prompt, steps, seed, cfg_scale, width, height, progress):
    print("Generating image...")
    pipe.to("cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with calculateDuration("Generating image"):
        # Generate image
        for img in pipe.flux_pipe_call_that_returns_an_iterable_of_images(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height,
            generator=generator,
            joint_attention_kwargs={"scale": 1.0},
            output_type="pil",
            good_vae=good_vae,
        ):
            # Yielding a tuple with image, seed, and a progress update
            yield img, seed, f"Generated image {img} with seed {seed}"
    return img

@spaces.GPU(duration=75)
def run_lora(prompt, cfg_scale, steps, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, randomize_seed, seed, width, height, loras_state, progress=gr.Progress(track_tqdm=True)):
    if not selected_indices:
        raise gr.Error("You must select at least one LoRA before proceeding.")

    selected_loras = [loras_state[idx] for idx in selected_indices]

    # Print the selected LoRAs
    print("Running with the following LoRAs:")
    for lora in selected_loras:
        print(f"- {lora['title']} from {lora['repo']} with scale {lora_scale_1 if selected_loras.index(lora) == 0 else lora_scale_2}")

    # Build the prompt with trigger words
    prepends = []
    appends = []
    for lora in selected_loras:
        trigger_word = lora.get('trigger_word', '')
        if trigger_word:
            if lora.get("trigger_position") == "prepend":
                prepends.append(trigger_word)
            else:
                appends.append(trigger_word)
    prompt_mash = " ".join(prepends + [prompt] + appends)
    print("Prompt Mash: ", prompt_mash)
    print("--Seed--:", seed)

    # Unload previous LoRA weights
    with calculateDuration("Unloading LoRA"):
        pipe.unload_lora_weights()
        
    print(pipe.get_active_adapters())

    # Load LoRA weights
    lora_names = []
    lora_weights = []
    with calculateDuration("Loading LoRA weights"):
        for idx, lora in enumerate(selected_loras):
            lora_name = f"lora_{idx}"
            lora_names.append(lora_name)
            print(f"Lora Name: {lora_name}")
            lora_weights.append(lora_scale_1 if idx == 0 else lora_scale_2)
            pipe.load_lora_weights(
                lora['repo'],
                weight_name=lora.get("weights"),
                low_cpu_mem_usage=True,
                adapter_name=lora_name,
            )
        print("Base Model:", base_model)
        print("Loaded LoRAs:", selected_indices)
        print("Adapter weights:", lora_weights)
        
        pipe.set_adapters(lora_names, adapter_weights=lora_weights)

    # Set random seed if required
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)

    # Generate image
    image_generator = generate_image(prompt_mash, steps, seed, cfg_scale, width, height, progress)
    step_counter = 0

    for image, seed, progress_update in image_generator:
        step_counter += 1
        progress_bar = f'<div class="progress-container"><div class="progress-bar" style="--current: {step_counter}; --total: {steps};"></div></div>'
        yield image, seed, gr.update(value=progress_bar, visible=True)

run_lora.zerogpu = True

def get_huggingface_safetensors(link, token=None):
    split_link = link.split("/")
    if len(split_link) == 2:
        model_card = ModelCard.load(link, use_auth_token=token)
        base_model = model_card.data.get("base_model")
        print(f"Base model: {base_model}")
        if base_model not in ["black-forest-labs/FLUX.1-dev", "black-forest-labs/FLUX.1-schnell"]:
            raise Exception("Not a FLUX LoRA!")
        image_path = model_card.data.get("widget", [{}])[0].get("output", {}).get("url", None)
        trigger_word = model_card.data.get("instance_prompt", "")
        image_url = f"https://huggingface.co/{link}/resolve/main/{image_path}" if image_path else None
        fs = HfFileSystem(token=token)
        safetensors_name = None
        try:
            list_of_files = fs.ls(link, detail=False)
            for file in list_of_files:
                if file.endswith(".safetensors"):
                    safetensors_name = file.split("/")[-1]
                if not image_url and file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    image_elements = file.split("/")
                    image_url = f"https://huggingface.co/{link}/resolve/main/{image_elements[-1]}"
        except Exception as e:
            print(e)
            raise gr.Error("Invalid Hugging Face repository with a *.safetensors LoRA")
        if not safetensors_name:
            raise gr.Error("No *.safetensors file found in the repository")
        return split_link[1], link, safetensors_name, trigger_word, image_url
    else:
        raise gr.Error("Invalid Hugging Face repository link")

def check_custom_model(link, token=None):
    if link.endswith(".safetensors"):
        title = os.path.basename(link)
        repo = link
        path = None
        trigger_word = ""
        image_url = None
        return title, repo, path, trigger_word, image_url
    elif link.startswith("https://"):
        if "huggingface.co" in link:
            link_split = link.split("huggingface.co/")
            return get_huggingface_safetensors(link_split[1], token=token)
        else:
            raise Exception("Unsupported URL")
    else:
        return get_huggingface_safetensors(link, token=token)

def update_history(new_image, history):
    """Updates the history gallery with the new image."""
    if history is None:
        history = []
    history.insert(0, new_image)
    return history

css = '''
#gen_btn{height: 100%}
#title{text-align: center}
#title h1{font-size: 2em; display:inline-flex; align-items:center}
#title img{width: 100px; margin-right: 0.25em}
#gallery .grid-wrap{height: 5vh}
#lora_list{background: var(--block-background-fill);padding: 0 1em .3em; font-size: 90%}
.custom_lora_card{margin-bottom: 1em}
.card_internal{display: flex;height: 100px;margin-top: .5em}
.card_internal img{margin-right: 1em}
.styler{--form-gap-width: 0px !important}
#progress{height:30px}
#progress .generating{display:none}
.progress-container {width: 100%;height: 30px;background-color: #f0f0f0;border-radius: 15px;overflow: hidden;margin-bottom: 20px}
.progress-bar {height: 100%;background-color: #4f46e5;width: calc(var(--current) / var(--total) * 100%);transition: width 0.5s ease-in-out}
#component-8, .button_total{height: 100%; align-self: stretch;}
#loaded_loras [data-testid="block-info"]{font-size:80%}
#custom_lora_structure{background: var(--block-background-fill)}
#custom_lora_btn{margin-top: auto;margin-bottom: 11px}
#random_btn{font-size: 300%}
#component-11{align-self: stretch;}
'''
font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"]
with gr.Blocks(theme=gr.themes.Soft(font=font), css=css, delete_cache=(128, 256)) as app:
    title = gr.HTML(
        """<h1><img src="https://huggingface.co/spaces/keltezaa/Celebrity_LoRa_Mix/resolve/main/solo-traveller_16875043.png" alt="LoRA">Celebrity_LoRa_Mix</h1>""",
        elem_id="title",
    )
    loras_state = gr.State(loras)
    selected_indices = gr.State([])
    trigger_word_display = gr.Markdown("", elem_id="trigger_word")

    with gr.Row():
        with gr.Column(scale=3):
            prompt = gr.Textbox(label="Prompt", lines=1, placeholder="Type a prompt after selecting a LoRA")
           
    with gr.Row(elem_id="loaded_loras"):
                   
        with gr.Column(scale=8):
            with gr.Row():
                with gr.Column(scale=0, min_width=50):
                    lora_image_1 = gr.Image(label="LoRA 1 Image", interactive=False, width=50, show_label=False, show_share_button=False, show_download_button=False, show_fullscreen_button=False, height=50)
                with gr.Column(scale=3, min_width=100):
                    selected_info_1 = gr.Markdown("Select a LoRA 1")
                with gr.Column(scale=5, min_width=50):
                    lora_scale_1 = gr.Slider(label="LoRA 1 Scale", minimum=0, maximum=3, step=0.05, value=0.5)
            with gr.Row():
                remove_button_1 = gr.Button("Remove", size="sm")
                
        with gr.Column(scale=8):
            with gr.Row():
                with gr.Column(scale=0, min_width=50):
                    lora_image_2 = gr.Image(label="LoRA 2 Image", interactive=False, width=50, show_label=False, show_share_button=False, show_download_button=False, show_fullscreen_button=False, height=50)
                with gr.Column(scale=3, min_width=100):
                    selected_info_2 = gr.Markdown("Select a LoRA 2")
                with gr.Column(scale=5, min_width=50):
                    lora_scale_2 = gr.Slider(label="LoRA 2 Scale", minimum=0, maximum=3, step=0.05, value=0.5)
            with gr.Row():
                remove_button_2 = gr.Button("Remove", size="sm")
            
        with gr.Column(scale=1,min_width=50):
            randomize_button = gr.Button("🎲", variant="secondary", scale=1, elem_id="random_btn")
                
    with gr.Row(elem_id="loaded_loras"):            
        with gr.Column(scale=8):
            with gr.Row():
                with gr.Column(scale=0, min_width=50):
                    lora_image_3 = gr.Image(label="LoRA 3 Image", interactive=False, width=50, show_label=False, show_share_button=False, show_download_button=False, show_fullscreen_button=False, height=50)
                with gr.Column(scale=3, min_width=100):
                    selected_info_3 = gr.Markdown("Select a LoRA 3")
                with gr.Column(scale=5, min_width=50):
                    lora_scale_3 = gr.Slider(label="LoRA 3 Scale", minimum=0, maximum=3, step=0.05, value=0.5)
            with gr.Row():
                remove_button_3 = gr.Button("Remove", size="sm")
        with gr.Column(scale=8):
            with gr.Row():
                with gr.Column(scale=0, min_width=50):
                    lora_image_4 = gr.Image(label="LoRA 4 Image", interactive=False, width=50, show_label=False, show_share_button=False, show_download_button=False, show_fullscreen_button=False, height=50)
                with gr.Column(scale=3, min_width=100):
                    selected_info_4 = gr.Markdown("Select a LoRA 4")
                with gr.Column(scale=5, min_width=150):
                    lora_scale_4 = gr.Slider(label="LoRA 4 Scale", minimum=0, maximum=3, step=0.05, value=0.5)
            with gr.Row():
                remove_button_4 = gr.Button("Remove", size="sm")

    with gr.Row():
        with gr.Accordion("Advanced Settings", open=True):
            #with gr.Row():
            #    input_image = gr.Image(label="Input image", type="filepath", show_share_button=False)
            #    image_strength = gr.Slider(label="Denoise Strength", info="Lower means more image influence", minimum=0.1, maximum=1.0, step=0.01, value=0.75)
            with gr.Column():
                with gr.Row():
                    cfg_scale = gr.Slider(label="CFG Scale", minimum=1, maximum=20, step=0.5, value=7.5)
                    steps = gr.Slider(label="Steps", minimum=1, maximum=50, step=1, value=28)

                with gr.Row():
                    width = gr.Slider(label="Width", minimum=256, maximum=1536, step=64, value=768)
                    height = gr.Slider(label="Height", minimum=256, maximum=1536, step=64, value=1024)

                with gr.Row():
                    randomize_seed = gr.Checkbox(True, label="Randomize seed")
                    seed = gr.Slider(label="Seed", minimum=0, maximum=MAX_SEED, step=1, value=0, randomize=True)

    with gr.Row():
        with gr.Column(scale=3):
            generate_button = gr.Button("Generate", variant="primary", elem_classes=["button_total"])
            
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row(elem_id="custom_lora_structure"):
                    custom_lora = gr.Textbox(label="Custom LoRA", info="LoRA Hugging Face path or *.safetensors public URL", placeholder="multimodalart/vintage-ads-flux", scale=3, min_width=150)
                    add_custom_lora_button = gr.Button("Add Custom LoRA", elem_id="custom_lora_btn", scale=2, min_width=150)
                remove_custom_lora_button = gr.Button("Remove Custom LoRA", visible=False)
                gr.Markdown("[Check the list of FLUX LoRAs](https://huggingface.co/models?other=base_model:adapter:black-forest-labs/FLUX.1-dev)", elem_id="lora_list")
            gallery = gr.Gallery(
                [(item["image"], item["title"]) for item in loras],
                label="Or pick from the gallery",
                allow_preview=False,
                columns=5,
                elem_id="gallery",
                show_share_button=False,
                interactive=False
            )
        with gr.Column():
            progress_bar = gr.Markdown(elem_id="progress", visible=False)
            result = gr.Image(label="Generated Image", interactive=False, show_share_button=False)
#            with gr.Accordion("History", open=False):
#                history_gallery = gr.Gallery(label="History", columns=6, object_fit="contain", interactive=False)  

    gallery.select(
        update_selection,
        inputs=[selected_indices, loras_state, width, height],
        outputs=[prompt, selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, width, height, lora_image_1, lora_image_2, lora_image_3, lora_image_4])
    remove_button_1.click(
        remove_lora_1,
        inputs=[selected_indices, loras_state],
        outputs=[selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4]
    )
    remove_button_2.click(
        remove_lora_2,
        inputs=[selected_indices, loras_state],
        outputs=[selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4]
    )
    remove_button_3.click(
        remove_lora_3,
        inputs=[selected_indices, loras_state],
        outputs=[selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4]
    )
    remove_button_4.click(
        remove_lora_4,
        inputs=[selected_indices, loras_state],
        outputs=[selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4]
    )
    randomize_button.click(
        randomize_loras,
        inputs=[selected_indices, loras_state],
        outputs=[selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4, prompt]
    )
    add_custom_lora_button.click(
        add_custom_lora,
        inputs=[custom_lora, selected_indices, loras_state, gallery],
        outputs=[loras_state, gallery, selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4]
    )

    remove_custom_lora_button.click(
        remove_custom_lora,
        inputs=[selected_indices, loras_state, gallery],
        outputs=[loras_state, gallery, selected_info_1, selected_info_2, selected_info_3, selected_info_4, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, lora_image_1, lora_image_2, lora_image_3, lora_image_4]
    )

    gr.on(
        triggers=[generate_button.click, prompt.submit],
        fn=run_lora,
        inputs=[prompt, cfg_scale, steps, selected_indices, lora_scale_1, lora_scale_2, lora_scale_3, lora_scale_4, randomize_seed, seed, width, height, loras_state],
        outputs=[result, seed, progress_bar]
    )#.then(
     #   fn=lambda x, history: update_history(x, history),
     #   inputs=[result, history_gallery],
     #   outputs=history_gallery,
    #)

app.queue()
app.launch()