#!/usr/bin/env python3
"""
Lumina - Main Entry Point

Usage:
  python run.py --mode train --config config/training.yaml
  python run.py --mode generate --prompt "Hello" --length 100
  python run.py --mode chat
  python run.py --mode api
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from backend.trainer import TerminalTrainer
from inference.pipeline import InferencePipeline
from utils.logger import setup_logger
from utils.device import get_device

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Lumina - Educational Transformer LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python run.py --mode train --config config/training.yaml
  
  # Generate text
  python run.py --mode generate --prompt "Once upon a time" --length 200
  
  # Interactive chat
  python run.py --mode chat --checkpoint checkpoints/model_latest.pt
  
  # Start API server
  python run.py --mode api --port 8000
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "generate", "chat", "api"],
        default="train",
        help="Mode to run: train, generate, chat, or api"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/training.yaml",
        help="Path to config YAML file (for train mode)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello",
        help="Prompt for generation (for generate mode)"
    )
    
    parser.add_argument(
        "--length",
        type=int,
        default=100,
        help="Length of generated text (for generate mode)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for generate/chat modes)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for API server (for api mode)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to use: cpu, cuda, or auto-detect"
    )
    
    args = parser.parse_args()
    
    # Detect device
    if args.device == "auto":
        device = get_device()
    else:
        device = args.device
    
    logger.info(f"Lumina - Educational Transformer LLM")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Device: {device}")
    
    try:
        if args.mode == "train":
            logger.info(f"Loading config from {args.config}")
            trainer = TerminalTrainer(config_path=args.config, device=device)
            trainer.run()
            
        elif args.mode == "generate":
            logger.info(f"Loading checkpoint: {args.checkpoint}")
            pipeline = InferencePipeline(
                checkpoint_path=args.checkpoint,
                device=device,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            result = pipeline.generate(
                prompt=args.prompt,
                max_length=args.length
            )
            print("\n" + "="*60)
            print("Generated Text:")
            print("="*60)
            print(result)
            print("="*60 + "\n")
            
        elif args.mode == "chat":
            logger.info(f"Starting interactive chat")
            pipeline = InferencePipeline(
                checkpoint_path=args.checkpoint,
                device=device,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p
            )
            pipeline.interactive_chat()
            
        elif args.mode == "api":
            logger.info(f"Starting API server on port {args.port}")
            import uvicorn
            from backend.api import app
            uvicorn.run(app, host="0.0.0.0", port=args.port)
    
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
