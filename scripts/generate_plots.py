import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from visualization.training_plots import plot_loss_curves
from pathlib import Path

def extract_losses(event_file):
    ea = EventAccumulator(event_file)
    ea.Reload()
    
    # Check available tags
    tags = ea.Tags()['scalars']
    
    g_losses = []
    d_losses = []
    
    # Note: We logged "Train/G_loss" and "Train/D_loss" in base_trainer.py
    if 'Train/G_loss' in tags:
        g_losses = [e.value for e in ea.Scalars('Train/G_loss')]
    if 'Train/D_loss' in tags:
        d_losses = [e.value for e in ea.Scalars('Train/D_loss')]
        
    return g_losses, d_losses

def main():
    output_root = Path("outputs")
    models = ["dcgan_anime", "wgan_gp_anime", "sagan_anime", "use_cmhsa_anime"]
    
    for model in models:
        model_dir = output_root / model
        log_dir = model_dir / "logs"
        report_dir = model_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        event_files = glob.glob(str(log_dir / "events.out.tfevents*"))
        if not event_files:
            print(f"No logs found for {model}")
            continue
            
        # Combine data if multiple files (though usually one per run)
        all_g = []
        all_d = []
        for f in sorted(event_files):
            g, d = extract_losses(f)
            all_g.extend(g)
            all_d.extend(d)
            
        if all_g and all_d:
            save_path = report_dir / "loss_curves.png"
            plot_loss_curves(
                all_g, 
                all_d, 
                save_path, 
                title=f"Loss Curves: {model.split('_')[0].upper()}"
            )
            print(f"Generated plot for {model} at {save_path}")

if __name__ == "__main__":
    main()
