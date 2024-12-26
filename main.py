from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from model import JEPAWorldModel
import argparse

# Parse arguments
parser = argparse.ArgumentParser(description='Evaluate using trained model')
parser.add_argument("-checkpoint", type=str, default='model_weights.pth')
parser.add_argument("-trained", type=str, default='true')
args = parser.parse_args()

def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def load_data(device):
    """Load all probing datasets."""
    data_path = "/scratch/DL24FA"

    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        train=False,
    )
    
    probe_val_wall_other_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall_other/val",
        probing=True,
        device=device,
        train=False,
    )

    probe_val_ds = {
        "normal": probe_val_normal_ds,
        "wall": probe_val_wall_ds,
        "wall_other": probe_val_wall_other_ds
    }

    return probe_train_ds, probe_val_ds

def load_expert_data(device):
    """Load expert validation datasets."""
    data_path = "/scratch/DL24FA"

    probe_train_expert_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_expert/train",
        probing=True,
        device=device,
        train=True,
    )

    probe_val_expert_ds = {
        "expert": create_wall_dataloader(
            data_path=f"{data_path}/probe_expert/val",
            probing=True,
            device=device,
            train=False,
        )
    }

    return probe_train_expert_ds, probe_val_expert_ds

def load_model():
    """Load trained model from checkpoint."""
    model = JEPAWorldModel()
    if args.trained == 'true':
        print(f"Loading checkpoint from: {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    model.eval()
    return model

def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    """Evaluate model using probing."""
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    prober = evaluator.train_pred_prober()
    avg_losses = evaluator.evaluate_all(prober=prober)

    # Print losses in desired format
    for probe_attr, loss in sorted(avg_losses.items()):
        if probe_attr == "normal":
            print(f"normal loss: {loss}")
        elif probe_attr == "wall":
            print(f"wall loss: {loss}")
        elif probe_attr == "wall_other":
            print(f"wall_other loss: {loss}")
        elif probe_attr == "expert":
            print(f"expert loss: {loss}")

if __name__ == "__main__":
    # Setup and evaluation
    device = get_device()
    model = load_model()
    model = model.to(device)
    
    # Print model parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal Trainable Parameters: {total_params:,}")

    # First run normal, wall, and wall_other evaluation
    print("\nEvaluating standard validation sets...")
    probe_train_ds, probe_val_ds = load_data(device)
    evaluate_model(device, model, probe_train_ds, probe_val_ds)

    # Then run expert evaluation
    print("\nEvaluating expert validation set...")
    probe_train_expert_ds, probe_val_expert_ds = load_expert_data(device)
    evaluate_model(device, model, probe_train_expert_ds, probe_val_expert_ds)