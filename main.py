"""Main entry point for Lumina"""

import argparse
import logging
import sys
from pathlib import Path

from backend.trainer import TerminalTrainer
from inference.pipeline import InferencePipeline
from utils.logger import setup_logger
from utils.device import get_device

logger = logging.getLogger(__name__)


def main():
    """
    Main entry point.
    """
    parser = argparse.ArgumentParser(
        description="Lumina - Educational Transformer Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python -m lumina train --config config.yaml --device cuda
  
  # Generate text from checkpoint
  python -m lumina generate --checkpoint model.pt --prompt "Hello"
  
  # Interactive chat
  python -m lumina chat --checkpoint model.pt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--config', type=str, default='config.yaml',
                             help='Path to configuration file')
    train_parser.add_argument('--device', type=str, default='auto',
                             help='Device to use (cpu, cuda, auto)')
    train_parser.add_argument('--resume', type=str, default=None,
                             help='Path to checkpoint to resume from')
    
    # Generation command
    gen_parser = subparsers.add_parser('generate', help='Generate text')
    gen_parser.add_argument('--checkpoint', type=str, required=True,
                           help='Path to model checkpoint')
    gen_parser.add_argument('--prompt', type=str, default='Once upon a time',
                           help='Prompt text')
    gen_parser.add_argument('--max-length', type=int, default=100,
                           help='Maximum generation length')
    gen_parser.add_argument('--temperature', type=float, default=0.7,
                           help='Sampling temperature')
    gen_parser.add_argument('--device', type=str, default='auto',
                           help='Device to use')
    
    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat')
    chat_parser.add_argument('--checkpoint', type=str, required=True,
                            help='Path to model checkpoint')
    chat_parser.add_argument('--device', type=str, default='auto',
                            help='Device to use')
    chat_parser.add_argument('--temperature', type=float, default=0.7,
                            help='Sampling temperature')
    
    # API command
    api_parser = subparsers.add_parser('api', help='Start FastAPI server')
    api_parser.add_argument('--checkpoint', type=str, default=None,
                           help='Path to model checkpoint to load')
    api_parser.add_argument('--host', type=str, default='0.0.0.0',
                           help='Host to bind to')
    api_parser.add_argument('--port', type=int, default=8000,
                           help='Port to bind to')
    api_parser.add_argument('--reload', action='store_true',
                           help='Enable auto-reload on code changes')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'generate':
        generate_command(args)
    elif args.command == 'chat':
        chat_command(args)
    elif args.command == 'api':
        api_command(args)
    else:
        parser.print_help()


def train_command(args):
    """
    Handle training command.
    """
    config_path = args.config
    
    if not Path(config_path).exists():
        logger.error(f"Configuration file not found: {config_path}")
        sys.exit(1)
    
    trainer = TerminalTrainer(config_path, device=args.device)
    trainer.run()


def generate_command(args):
    """
    Handle text generation command.
    """
    setup_logger(__name__, level='INFO')
    
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    logger.info("Loading model...")
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        device=args.device,
        temperature=args.temperature
    )
    
    logger.info("Generating text...")
    print("\n" + "="*60)
    print("Text Generation")
    print("="*60 + "\n")
    
    generated_text = pipeline.generate(
        prompt=args.prompt,
        max_length=args.max_length,
        strategy='temperature'
    )
    
    print(f"Prompt: {args.prompt}")
    print(f"\nGenerated: {generated_text}\n")


def chat_command(args):
    """
    Handle interactive chat command.
    """
    setup_logger(__name__, level='INFO')
    
    checkpoint_path = args.checkpoint
    if not Path(checkpoint_path).exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    logger.info("Loading model...")
    pipeline = InferencePipeline(
        checkpoint_path=checkpoint_path,
        device=args.device,
        temperature=args.temperature
    )
    
    pipeline.interactive_chat()


def api_command(args):
    """
    Handle API server command.
    """
    import uvicorn
    from backend.api import app
    
    logger.info(f"Starting Lumina API server on {args.host}:{args.port}")
    
    if args.checkpoint:
        from backend.api import pipeline
        logger.info(f"Loading checkpoint: {args.checkpoint}")
        pipeline.load_model(args.checkpoint)
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == '__main__':
    main()
