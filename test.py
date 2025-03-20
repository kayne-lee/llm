from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the model and tokenizer name
model_name = 'microsoft/phi-2'
hf_token = os.getenv('HF')

# Set the directory where the model will be saved
local_model_dir = './local_phi2_model/'

# Authenticate and download model
model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)

# Save model and tokenizer locally
model.save_pretrained(local_model_dir)
tokenizer.save_pretrained(local_model_dir)

print(f"Model and tokenizer saved locally at {local_model_dir}")


print("Model and tokenizer loaded successfully!")
