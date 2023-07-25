from transformers import pipeline, TextGenerationPipeline, GPT2LMHeadModel, AutoTokenizer
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionPipeline, DPMSolverMultistepScheduler


def load_models():
    # Load Models I trained from my google drive :
    story_model_folder_path = "/content/drive/MyDrive/Generative_model/Generative-AI-Project/My-Tuned-Story-Generator-model"
    stable_diffusion_base_model = "/content/drive/MyDrive/Generative_model/Generative-AI-Project/My-Tuned-Stable-diffusion-txt2img/Base_model"
    stable_diffusion_model_path = "/content/drive/MyDrive/Generative_model/Generative-AI-Project/My-Tuned-Stable-diffusion-txt2img/My_Tuned_Model"
    
    story_loaded_model = GPT2LMHeadModel.from_pretrained(story_model_folder_path)
    story_loaded_tokenizer = AutoTokenizer.from_pretrained(story_model_folder_path)
    loaded_story_generator = TextGenerationPipeline(model=story_loaded_model, tokenizer=story_loaded_tokenizer)
    
    pipe = StableDiffusionPipeline.from_pretrained(stable_diffusion_base_model, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    pipe.unet.load_attn_procs(stable_diffusion_model_path)
    pipe.to("cuda")

    return loaded_story_generator, pipe