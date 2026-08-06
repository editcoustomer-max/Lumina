"""Training backend with terminal interface"""

import torch
import logging
from pathlib import Path
from typing import Optional

from utils.logger import setup_logger
from utils.helpers import load_yaml, ensure_dir
from data.dataset import DatasetLoader
from data.loader import DataLoaderFactory
from tokenizer.tokenizer import CharacterTokenizer, BPETokenizer
from training.trainer import Trainer

logger = logging.getLogger(__name__)


class TerminalTrainer:
    """
    Training interface with terminal logging.
    """
    
    def __init__(self, config_path: str, device: str = 'cpu'):
        """
        Initialize terminal trainer.
        
        Args:
            config_path: Path to configuration YAML file
            device: Device to train on
        """
        # Load configuration
        self.config = load_yaml(config_path)
        self.device = device
        
        # Setup directories
        ensure_dir(self.config['checkpointing']['checkpoint_dir'])
        ensure_dir(self.config['logging'].get('log_file', 'logs/').rsplit('/', 1)[0])
        
        # Setup logger
        log_file = self.config['logging'].get('log_file')
        log_level = self.config['logging'].get('level', 'INFO')
        setup_logger(__name__, log_file=log_file, level=log_level)
        
        logger.info("="*60)
        logger.info("Lumina - Educational Transformer LLM")
        logger.info("="*60)
        logger.info(f"Configuration: {config_path}")
        logger.info(f"Device: {device}")
        
        self.trainer = None
    
    def prepare_data(self):
        """
        Load and prepare training data.
        """
        logger.info("\n" + "="*60)
        logger.info("Preparing Data")
        logger.info("="*60)
        
        # Load datasets
        loader = DatasetLoader(self.config['dataset']['data_dir'])
        text = loader.load_datasets()
        
        if not text:
            logger.error("No data loaded. Please check dataset directory.")
            return None
        
        logger.info(f"Dataset stats: {loader.get_stats()}")
        
        return text, loader
    
    def build_tokenizer(self, text: str) -> object:
        """
        Build tokenizer from data.
        
        Args:
            text: Training text
        
        Returns:
            Tokenizer instance
        """
        logger.info("\n" + "="*60)
        logger.info("Building Tokenizer")
        logger.info("="*60)
        
        tokenizer_config = self.config['tokenizer']
        tokenizer_type = tokenizer_config.get('type', 'character')
        
        if tokenizer_type == 'character':
            tokenizer = CharacterTokenizer()
            tokenizer.build_vocab(text)
        elif tokenizer_type == 'bpe':
            tokenizer = BPETokenizer(vocab_size=tokenizer_config.get('vocab_size', 1000))
            tokenizer.build_vocab(text)
        else:
            logger.error(f"Unknown tokenizer type: {tokenizer_type}")
            return None
        
        logger.info(f"Built {tokenizer_type} tokenizer with {tokenizer.get_vocab_size()} tokens")
        
        # Save tokenizer
        if tokenizer_config.get('save_tokenizer'):
            tokenizer_path = Path(self.config['checkpointing']['checkpoint_dir']) / f"tokenizer_{tokenizer_type}.json"
            tokenizer.save(str(tokenizer_path))
            logger.info(f"Saved tokenizer to {tokenizer_path}")
        
        return tokenizer
    
    def create_data_loaders(self, text: str, tokenizer) -> tuple:
        """
        Create training and validation data loaders.
        
        Args:
            text: Training text
            tokenizer: Tokenizer instance
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        logger.info("\n" + "="*60)
        logger.info("Creating Data Loaders")
        logger.info("="*60)
        
        dataset_config = self.config['dataset']
        training_config = self.config['training']
        
        train_loader, val_loader = DataLoaderFactory.create_loader(
            text=text,
            tokenizer=tokenizer,
            seq_length=dataset_config.get('seq_length', 512),
            batch_size=training_config.get('batch_size', 32),
            shuffle=dataset_config.get('shuffle', True),
            train_split=dataset_config.get('train_split', 0.9),
        )
        
        logger.info(f"Created data loaders")
        
        return train_loader, val_loader
    
    def run(self):
        """
        Run training pipeline.
        """
        try:
            # Prepare data
            data_result = self.prepare_data()
            if not data_result:
                return
            
            text, loader = data_result
            
            # Build tokenizer
            tokenizer = self.build_tokenizer(text)
            if not tokenizer:
                return
            
            # Update config with actual vocab size
            self.config['model']['vocab_size'] = tokenizer.get_vocab_size()
            
            # Create data loaders
            train_loader, val_loader = self.create_data_loaders(text, tokenizer)
            
            # Initialize trainer
            logger.info("\n" + "="*60)
            logger.info("Initializing Model")
            logger.info("="*60)
            
            self.trainer = Trainer(self.config, device=self.device)
            
            # Train
            logger.info("\n" + "="*60)
            logger.info("Starting Training")
            logger.info("="*60)
            
            self.trainer.fit(train_loader, val_loader)
            
            logger.info("\n" + "="*60)
            logger.info("Training Complete")
            logger.info("="*60)
        
        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            raise
