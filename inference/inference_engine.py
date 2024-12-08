
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class InferenceEngine:
    def __init__(self, model_name="gpt2", device="cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None

    def load_model(self):
        try:
            print(f"Loading model {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            self.model.to(self.device)
            print(f"Model {self.model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def run_inference(self, prompt, max_length=50, temperature=0.7, top_k=50):
        if not self.model or not self.tokenizer:
            raise ValueError("Model and tokenizer must be loaded before running inference.")

        print(f"Running inference on prompt: {prompt}")
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True
            )
            decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Inference output: {decoded_output}")
            return decoded_output
        except Exception as e:
            print(f"Error during inference: {e}")
            raise

if __name__ == "__main__":
    engine = InferenceEngine(model_name="gpt2", device="cpu")
    engine.load_model()
    prompt = "Once upon a time in a faraway land"
    output = engine.run_inference(prompt, max_length=100)
    print(f"Generated text: {output}")
