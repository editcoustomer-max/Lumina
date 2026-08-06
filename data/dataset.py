"""Dataset loading and management"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import json
from data.preprocessor import TextPreprocessor

logger = logging.getLogger(__name__)


class DatasetLoader:
    """
    Load text datasets from files and directories.
    """
    
    def __init__(self, data_dir: str = "datasets/", encoding: str = "utf-8"):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Root directory containing datasets
            encoding: Text encoding (default: utf-8)
        """
        self.data_dir = Path(data_dir)
        self.encoding = encoding
        self.texts = []
        self.file_list = []
        self.stats = {}
    
    def scan_datasets(self) -> Dict[str, List[Path]]:
        """
        Scan dataset directory for .txt files.
        
        Returns:
            Dictionary mapping category to list of file paths
        """
        if not self.data_dir.exists():
            logger.warning(f"Dataset directory not found: {self.data_dir}")
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return {}
        
        datasets = {}
        
        # Scan all subdirectories
        for category_dir in self.data_dir.iterdir():
            if category_dir.is_dir():
                txt_files = list(category_dir.glob('*.txt'))
                if txt_files:
                    datasets[category_dir.name] = txt_files
                    logger.info(f"Found {len(txt_files)} files in category: {category_dir.name}")
        
        return datasets
    
    def load_datasets(self, max_files: Optional[int] = None) -> str:
        """
        Load all datasets into memory.
        
        Args:
            max_files: Maximum number of files to load (None = no limit)
        
        Returns:
            Concatenated text from all files
        """
        self.texts = []
        self.file_list = []
        
        datasets = self.scan_datasets()
        file_count = 0
        
        for category, files in datasets.items():
            for filepath in files:
                if max_files and file_count >= max_files:
                    logger.info(f"Reached maximum file limit: {max_files}")
                    break
                
                try:
                    with open(filepath, 'r', encoding=self.encoding) as f:
                        text = f.read()
                    
                    self.texts.append(text)
                    self.file_list.append(str(filepath))
                    file_count += 1
                    
                    logger.debug(f"Loaded: {filepath}")
                
                except Exception as e:
                    logger.error(f"Error loading {filepath}: {e}")
        
        # Concatenate all texts
        full_text = "\n\n".join(self.texts)
        
        logger.info(f"Loaded {len(self.texts)} files from {len(datasets)} categories")
        self._compute_stats(full_text)
        
        return full_text
    
    def load_from_directory(self, directory: str, recursive: bool = True) -> str:
        """
        Load all .txt files from a specific directory.
        
        Args:
            directory: Path to directory
            recursive: Whether to search recursively
        
        Returns:
            Concatenated text from all files
        """
        dir_path = Path(directory)
        self.texts = []
        self.file_list = []
        
        if recursive:
            txt_files = list(dir_path.rglob('*.txt'))
        else:
            txt_files = list(dir_path.glob('*.txt'))
        
        for filepath in txt_files:
            try:
                with open(filepath, 'r', encoding=self.encoding) as f:
                    text = f.read()
                self.texts.append(text)
                self.file_list.append(str(filepath))
                logger.debug(f"Loaded: {filepath}")
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        full_text = "\n\n".join(self.texts)
        logger.info(f"Loaded {len(self.texts)} files from {directory}")
        self._compute_stats(full_text)
        
        return full_text
    
    def _compute_stats(self, text: str):
        """
        Compute statistics about the loaded text.
        
        Args:
            text: Text to analyze
        """
        self.stats = {
            'num_files': len(self.file_list),
            'characters': len(text),
            'words': len(text.split()),
            'lines': len(text.split('\n')),
            'sentences': len(TextPreprocessor.split_into_sentences(text)),
        }
        
        logger.info(f"Dataset stats: {self.stats}")
    
    def get_stats(self) -> Dict:
        """
        Get dataset statistics.
        
        Returns:
            Dictionary with dataset statistics
        """
        return self.stats
    
    def get_file_list(self) -> List[str]:
        """
        Get list of loaded files.
        
        Returns:
            List of file paths
        """
        return self.file_list
