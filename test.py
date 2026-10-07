import torch
from Diffusion_Framework_For_Earthquake_Waveform_Generation.model import DiffusionModelConfig, HybridMambaDiffusionModel
from scheduler import DDPMScheduler


def testing_model(model, scheduler, cond_batches, seq_length, device):
    model.eval()
    predictions = []
    in_channels = model.config.in_channels

    with torch.no_grad():
        for cond in cond_batches:
            cond = cond.to(device)
            batch_size = cond.shape[0]

            x = torch.randn(
                (batch_size, in_channels, seq_length), device=device)

            for i in reversed(range(scheduler.num_timesteps)):
                t = torch.full((batch_size,), i,
                               device=device, dtype=torch.long)
                noise_pred = model(x, t, cond)
                x = scheduler.step(noise_pred, i, x)

            predictions.append(x.cpu())

    return torch.cat(predictions, dim=0)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = DiffusionModelConfig(
        in_channels=3, cond_channels=7, hidden_channels=128, d_state=128, d_conv=16, expand=2)
    model = HybridMambaDiffusionModel(config).to(device)

    model.load_state_dict(torch.load(
        "mamba_diffusion_weights.pth", map_location=device))
    model.eval()

    scheduler = DDPMScheduler(num_timesteps=1000, device=device)

    test_dataloader = ...  # test data add kar
    print("Starting generation...")
    generated_samples = testing_model(
        model, scheduler, test_dataloader, seq_length=256, device=device)

    torch.save(generated_samples, "generated_output.pt")
    print("Saved generated samples!")


if __name__ == "__main__":
    main()
