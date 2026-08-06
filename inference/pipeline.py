"""Inference pipeline for text generation"""

import torch
import logging
from typing import Optional, List
from model.transformer import TransformerLM
from model.checkpoint import CheckpointManager
from tokenizer.tokenizer import CharacterTokenizer, BPETokenizer
from utils.device import get_device

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Pipeline for text generation and inference.
    """
    
    def __init__(self, checkpoint_path: Optional[str] = None, device: str = 'auto',
                 temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9):
        """
        Initialize inference pipeline.
        
        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use (cpu, cuda, auto)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
        """
        self.device = get_device(device)
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.model = None
        self.tokenizer = None
        
        # Load model if checkpoint provided
        if checkpoint_path:
            self.load_model(checkpoint_path)
    
    def load_model(self, checkpoint_path: str):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint
        """
        self.model = CheckpointManager.load_model_from_checkpoint(
            checkpoint_path, device=self.device
        )
        logger.info(f"Loaded model from {checkpoint_path}")
    
    def set_tokenizer(self, tokenizer):
        """
        Set tokenizer for the pipeline.
        
        Args:
            tokenizer: Tokenizer instance
        """
        self.tokenizer = tokenizer
    
    @torch.no_grad()
    def generate(self, prompt: str, max_length: int = 100, 
                strategy: str = 'top_p') -> str:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum generation length
            strategy: Sampling strategy (greedy, temperature, top_k, top_p)
        
        Returns:
            Generated text
        """
        if self.model is None:
            logger.error("Model not loaded")
            return prompt
        
        if self.tokenizer is None:
            logger.error("Tokenizer not set")
            return prompt
        
        # Tokenize prompt
        prompt_ids = self.tokenizer.encode(prompt)
        prompt_ids = torch.tensor(prompt_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        
        current_ids = prompt_ids
        generated_ids = prompt_ids.clone()
        
        # Generate tokens
        for _ in range(max_length):
            # Get model prediction
            logits = self.model(current_ids)
            next_token_logits = logits[0, -1, :]  # Last token logits
            
            # Apply temperature
            if self.temperature != 0:
                next_token_logits = next_token_logits / self.temperature
            
            # Sampling strategy
            if strategy == 'greedy':
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            elif strategy == 'temperature':
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
            
            elif strategy == 'top_k':
                top_k_logits, top_k_indices = torch.topk(next_token_logits, self.top_k)
                top_k_probs = torch.softmax(top_k_logits, dim=-1)
                top_k_id = torch.multinomial(top_k_probs, num_samples=1)
                next_token_id = top_k_indices[top_k_id]
            
            elif strategy == 'top_p':
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Keep tokens with cumulative probability <= top_p
                sorted_indices_to_keep = cumsum_probs <= self.top_p
                sorted_indices_to_keep[0] = True  # Always keep at least one token
                
                sorted_probs[~sorted_indices_to_keep] = 0
                sorted_probs = sorted_probs / sorted_probs.sum()
                
                top_p_id = torch.multinomial(sorted_probs, num_samples=1)
                next_token_id = sorted_indices[top_p_id]
            
            else:
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            
            # Add to generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id.unsqueeze(0)], dim=1)
            current_ids = generated_ids[:, -1:] if generated_ids.size(1) > 512 else generated_ids
        
        # Decode generated text
        generated_ids_list = generated_ids[0].cpu().tolist()
        generated_text = self.tokenizer.decode(generated_ids_list)
        
        return generated_text
    
    def interactive_chat(self):
        """
        Start interactive chat session.
        """
        if self.model is None:
            logger.error("Model not loaded")
            return
        
        logger.info("Starting interactive chat (type 'quit' to exit)")
        print("\n" + "="*60)
        print("Interactive Chat - Type 'quit' to exit")
        print("="*60 + "\n")
        
        while True:
            try:
                prompt = input("You: ").strip()
                
                if prompt.lower() == 'quit':
                    logger.info("Exiting chat")
                    break
                
                if not prompt:
                    continue
                
                # Generate response
                response = self.generate(prompt, max_length=100)
                print(f"\nBot: {response}\n")
            
            except KeyboardInterrupt:
                logger.info("Chat interrupted")
                break
            except Exception as e:
                logger.error(f"Error during generation: {e}")
                continue
