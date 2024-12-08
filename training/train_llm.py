import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_llm(model_name='gpt2', data_path='data/text_data.txt', epochs=3, batch_size=8, lr=2e-5):
    logger.info("Loading tokenizer and model...")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    logger.info("Loading dataset...")
    with open(data_path, 'r') as f:
        text_data = f.read()
    tokenized_data = tokenizer(text_data, return_tensors="pt", truncation=True, padding=True, max_length=512)

    logger.info("Preparing optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

    logger.info("Starting training loop...")
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(**tokenized_data, labels=tokenized_data['input_ids'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        logger.info(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    logger.info("Saving model...")
    model.save_pretrained('trained_llm_model')
    tokenizer.save_pretrained('trained_llm_model')
    logger.info("Training complete and model saved.")

if __name__ == "__main__":
    train_llm()
