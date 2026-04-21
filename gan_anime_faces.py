"""Main entry point for Anime Face GAN project."""

import argparse
import logging
from pathlib import Path
import yaml

import torch
from torch.utils.data import DataLoader

from configs.base_config import BaseConfig
from configs.dcgan_config import DCGANConfig
from configs.wgan_gp_config import WGANGPConfig
from configs.sagan_config import SAGANConfig
from configs.use_cmhsa_config import USECMHSAConfig
from configs.stylegan2_config import StyleGAN2Config

from data.dataset import AnimeDataset
from data.augmentation import get_transform
from models.factory import create_model
from trainers.factory import create_trainer
from utils.logger import setup_logger
from utils.seed import set_seed
from utils.device import get_device

CONFIG_MAP = {
    "dcgan": DCGANConfig,
    "wgan_gp": WGANGPConfig,
    "sagan": SAGANConfig,
    "use_cmhsa": USECMHSAConfig,
    "stylegan2": StyleGAN2Config,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Anime Face Generation using GANs")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a GAN model")
    train_parser.add_argument(
        "--model", 
        type=str, 
        required=True,
        choices=list(CONFIG_MAP.keys()),
        help="Model architecture to train"
    )
    train_parser.add_argument("--epochs", type=int, help="Override number of epochs")
    train_parser.add_argument("--batch-size", type=int, help="Override batch size")
    train_parser.add_argument("--quick-test", action="store_true", help="Run a fast 2-epoch test")
    train_parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")
    
    # Eval command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained models")
    eval_parser.add_argument(
        "--model", 
        type=str, 
        choices=list(CONFIG_MAP.keys()),
        help="Specific model architecture to evaluate"
    )
    eval_parser.add_argument("--all-models", action="store_true", help="Evaluate all trained models")
    eval_parser.add_argument("--metrics", type=str, default="fid,is", help="Metrics to compute (comma separated)")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.command == "train":
        # 1. Setup config
        ConfigClass = CONFIG_MAP[args.model]
        config = ConfigClass()
        
        # Overrides
        if args.epochs:
            config.num_epochs = args.epochs
        if args.batch_size:
            config.batch_size = args.batch_size
        if args.resume:
            config.resume_checkpoint = args.resume
            
        if args.quick_test:
            config.num_epochs = 2
            config.batch_size = 16
            config.eval_interval = 1
            config.checkpoint_interval = 1
            config.sample_interval = 1
            
        config.create_directories()
            
        # 2. Setup Environment
        logger = setup_logger(
            name="anime_gan",
            log_dir=config.log_dir
        )
        logger.info(f"Starting {config.experiment_name} training")
        
        set_seed(config.seed)
        device = get_device(config.device)
        
        # 3. Setup Data
        logger.info(f"Loading data from {config.data_dir}")
        transform = get_transform(
            image_size=config.image_size, 
            is_stylegan=(config.model_type == "stylegan2")
        )
        
        dataset = AnimeDataset(
            root_dir=config.data_dir,
            transform=transform
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        logger.info(f"Dataset size: {len(dataset)} | Batches per epoch: {len(dataloader)}")
        
        if config.model_type == "stylegan2":
            logger.error("StyleGAN2 requires custom training runner using official repo. Not implemented in this basic script.")
            return
            
        # 4. Setup Model
        logger.info(f"Initializing {config.model_type} architecture")
        generator, discriminator = create_model(config)
        generator = generator.to(device)
        discriminator = discriminator.to(device)
        
        # 5. Setup Trainer
        trainer = create_trainer(
            generator=generator,
            discriminator=discriminator,
            dataloader=dataloader,
            config=config,
            device=device
        )
        
        # Resume if specified
        if config.resume_checkpoint:
            logger.info(f"Resuming from {config.resume_checkpoint}")
            trainer.load_checkpoint(config.resume_checkpoint)
            
        # 6. Train
        trainer.train()
        
    elif args.command == "evaluate":
        logger = setup_logger("evaluator", log_dir=Path("/scratch/nishanth.r/gan_proj/outputs/eval_logs"))
        from evaluation.evaluator import Evaluator
        
        models_to_eval = list(CONFIG_MAP.keys()) if args.all_models else [args.model]
        
        device = get_device("cuda")
        metrics_list = args.metrics.split(",")
        
        for model_name in models_to_eval:
            if model_name is None or model_name == "stylegan2":
                continue
                
            logger.info(f"Evaluating {model_name}...")
            ConfigClass = CONFIG_MAP[model_name]
            config = ConfigClass()
            
            # Load Generator
            generator, _ = create_model(config)
            generator = generator.to(device)
            
            latest_ckpt = config.checkpoint_dir / "best_model.pt"
            if not latest_ckpt.exists():
                latest_ckpt = config.checkpoint_dir / "latest.pt"
                
            if latest_ckpt.exists():
                ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
                generator.load_state_dict(ckpt["generator_state_dict"])
            else:
                logger.warning(f"No checkpoint found for {model_name} at {latest_ckpt}")
                continue
                
            evaluator = Evaluator(config=config, generator=generator, device=device)
            evaluator.evaluate_all(metrics=metrics_list)
            evaluator.print_summary()


if __name__ == "__main__":
    main()
