import torch
from transformers import pipeline

# Check CUDA availability
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# Initialize the pipeline for text generation
model_id = "microsoft/phi-4"
phi_pipeline = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.bfloat16,
    device=0,  # Use RTX 4090 (cuda:0)
)

# Check pipeline device
print(f"Pipeline is running on device: {phi_pipeline.device}")

# Define a wrapper to simulate the `invoke` method
class LLMResponse:
    def __init__(self, content):
        self.content = content

class LLMWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def invoke(self, prompt):
        response = self.pipeline(prompt, max_new_tokens=128)
        return LLMResponse(content=response[0]["generated_text"])  # Return response with .content

# Wrap the pipeline
llm = LLMWrapper(phi_pipeline)

# Define the prompt
prompt = [
    {"role": "system", "content": "You are a medieval knight and must provide explanations to modern people."},
    {"role": "user", "content": "How should I explain the Internet?"},
]

# Invoke the model and get the response
response = llm.invoke(prompt).content
print(response)

