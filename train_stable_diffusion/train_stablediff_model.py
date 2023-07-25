import os
import subprocess
import argparse

## Note that you will need to install some dependencies before running this on colab

def train_model(max_train_steps):
    os.environ["MODEL_NAME"] = "CompVis/stable-diffusion-v1-4"
    os.environ["dataset_name"] = "Norod78/cartoon-blip-captions"

    subprocess.run([
        "accelerate", "launch", 
        "/content/diffusers/examples/text_to_image/train_text_to_image_lora.py",
        "--pretrained_model_name_or_path=" + os.environ["MODEL_NAME"],
        "--dataset_name=" + os.environ["dataset_name"],
        "--resolution=512", "--center_crop", "--random_flip",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--gradient_checkpointing",
        "--mixed_precision=fp16",
        "--max_train_steps=" + str(max_train_steps),
        "--learning_rate=1e-05",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--validation_promp=alien came to earth",
        "--output_dir=saved_tunned_model_new"
    ])

    subprocess.run(["zip", "-r", "saved_tunned_model.zip", "/content/saved_tunned_model"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_steps", type=int, default=5000, help="Maximum number of training steps")
    args = parser.parse_args()

    train_model(args.max_train_steps)


## to start training write :
## python train_stablediff_model.py --max_train_steps 5000   "you can change number of max_train_steps"
