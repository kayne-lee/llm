from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pdfplumber

model_name = "microsoft/phi-2"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and move to device
model = AutoModelForCausalLM.from_pretrained(model_name, local_files_only=True).to(device)

print("Model loaded successfully!")

pdfPath = "Syllabus2025.pdf"

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"  # Extract text from each page
    return text

pdf_text = extract_text_from_pdf(pdfPath)

prompt = (
    "Extract all assignments, tests, midterms, and exams from the syllabus. "
    "Return ONLY a **valid JSON object** with an 'assignments' array. "
    "Each item should have 'title', 'weight', and 'dueDate' (ISO format, default '2024-12-01T23:59' if missing). "
    "If no assessments are found, return an empty JSON object. "
    "Input: "
)




full_prompt = prompt + "Input: " + pdf_text
# Tokenize input and move tensors to the same device as the model
inputs = tokenizer(full_prompt, return_tensors="pt").to(device)

# Generate response
with torch.no_grad():
    output = model.generate(**inputs, max_new_tokens=200)

# Decode output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nGenerated Text:\n", generated_text)
