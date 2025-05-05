#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main entry point for BotChatAI - Stock Market AI Advisor
"""

import os
import sys
import logging
import tkinter as tk
import json
import time
import traceback
from datetime import datetime
import argparse
from pathlib import Path

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Initialize logging
from core.utils.logging_config import configure_logging, get_logger
logger = get_logger("main")

def load_config(config_path=None):
    """
    Load configuration from file
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    # Default config path
    if config_path is None:
        config_path = os.path.join(parent_dir, "config", "default_config.json")
    
    try:
        # Check if config file exists
        if not os.path.exists(config_path):
            logger.warning(f"Config file {config_path} not found, using internal defaults")
            return {}
        
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

def setup_environment(config):
    """
    Setup environment variables based on configuration
    
    Args:
        config: Configuration dictionary
    """
    # Create necessary directories
    dirs_to_create = [
        os.path.join(parent_dir, "logs"),
        os.path.join(parent_dir, "models"),
        os.path.join(parent_dir, "results"),
        os.path.join(parent_dir, "cache")
    ]
    
    for directory in dirs_to_create:
        os.makedirs(directory, exist_ok=True)
        logger.debug(f"Ensured directory exists: {directory}")
    
    # Set timezone
    os.environ['TZ'] = config.get('app', {}).get('timezone', 'Asia/Ho_Chi_Minh')
    
    # Set log level
    log_level = config.get('app', {}).get('log_level', 'INFO')
    logging.getLogger().setLevel(getattr(logging, log_level))

def display_startup_info(config):
    """Display startup information"""
    version = config.get('version', '1.0')
    app_name = config.get('app', {}).get('name', 'BotChatAI')
    
    logger.info(f"Starting {app_name} v{version}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info(f"Log directory: {os.path.join(parent_dir, 'logs')}")
    
    # Display config summary
    log_config_summary(config)

def log_config_summary(config):
    """Log a summary of the configuration"""
    try:
        ai_providers = list(config.get('ai', {}).get('providers', {}).keys())
        enabled_providers = [p for p in ai_providers if config.get('ai', {}).get('providers', {}).get(p, {}).get('enabled', False)]
        
        ml_model = config.get('ml_model', {}).get('model_name', 'unknown')
        ensemble = "enabled" if config.get('ensemble', {}).get('enabled', False) else "disabled"
        
        logger.info(f"AI Providers: {', '.join(enabled_providers)}")
        logger.info(f"ML Model: {ml_model}")
        logger.info(f"Ensemble Learning: {ensemble}")
        logger.info(f"Data Source: {config.get('data', {}).get('default_source', 'vnstock')}")
    except Exception as e:
        logger.error(f"Error logging config summary: {str(e)}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='BotChatAI - Stock Market AI Advisor')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    return parser.parse_args()

def main():
    """Main entry point"""
    start_time = time.time()
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        config = load_config(args.config)
        
        # Override debug mode from command line
        if args.debug:
            config.setdefault('app', {})['debug_mode'] = True
            logging.getLogger().setLevel(logging.DEBUG)
        
        # Setup environment
        setup_environment(config)
        
        # Display startup information
        display_startup_info(config)
        
        # Initialize tkinter
        root = tk.Tk()
        
        # Set window icon
        icon_path = os.path.join(parent_dir, "assets", "app_icon.ico")
        if os.path.exists(icon_path):
            root.iconbitmap(icon_path)
        
        # Optimize database if needed
        from core.data.db import DBManager
        db = DBManager()
        if config.get('database', {}).get('optimize_on_startup', False):
            logger.info("Optimizing database...")
            db.optimize_database()
        
        # Import GUI after environment setup
        from core.chatbot.chatbot import ChatbotGUI
        
        # Create GUI with config
        app = ChatbotGUI(root, config)
        
        # Start main loop
        logger.info(f"Application startup completed in {time.time() - start_time:.2f}s")
        root.mainloop()
        
    except Exception as e:
        logger.error(f"Critical error during startup: {str(e)}")
        logger.error(traceback.format_exc())
        
        # Show error dialog
        try:
            import tkinter.messagebox as messagebox
            messagebox.showerror("Startup Error", f"Critical error during startup:\n{str(e)}")
        except:
            print(f"Critical error: {str(e)}")
        
        sys.exit(1)

if __name__ == "__main__":
    main() 