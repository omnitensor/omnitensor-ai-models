import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class TransformerModel:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        print(f"Loading model and tokenizer for {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Model and tokenizer loaded successfully.")

    def infer(self, prompt: str, max_length: int = 50) -> str:
        """
        Generate text based on a prompt using the loaded transformer model.
        
        Args:
            prompt (str): The input text to prompt the model.
            max_length (int): Maximum length of the generated text.

        Returns:
            str: Generated text.
        """
        print(f"Generating text for prompt: {prompt}")
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1, do_sample=True)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")
        return generated_text

    def fine_tune(self, dataset, epochs: int = 3):
        """
        //to do
        """
        print(f"Fine-tuning {self.model_name} on custom dataset for {epochs} epochs...")
        # Note: Implement training logic with PyTorch and Hugging Face Trainer API.
        print("Fine-tuning complete (//).")

# Example usage:
if __name__ == "__main__":
    model = TransformerModel("gpt2")
    print(model.infer("Once upon a time"))
