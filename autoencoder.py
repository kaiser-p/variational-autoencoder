from __future__ import annotations

import argparse
from typing import Tuple
from pathlib import Path
import tqdm

import torch
import torchvision
import torch.utils.tensorboard
import torch.utils.data
from models import SimpleConvolutionalAE


def train(args: argparse.Namespace):
    train_dir = args.working_dir / "train"
    logs_dir = args.working_dir / "train" / "summary"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleConvolutionalAE(img_dim=args.img_size, bottleneck_dim=1)
    model.to(device)
    print(model)

    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)),  # TODO: Only resize the canvas
        torchvision.transforms.ToTensor()
    ])

    train_data_dir = args.working_dir / f"dataset_{args.dataset}"
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=image_transforms)

    reconstruction_loss_function = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)

    print("Starting training ...")
    with torch.utils.tensorboard.SummaryWriter(log_dir=logs_dir) as summary_writer:
        global_step = 0
        for epoch in range(args.num_epochs):
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, num_workers=0, shuffle=True)
            for local_step, (data, target) in enumerate(train_loader):
                global_step += 1

                optimizer.zero_grad()
                output = model.forward(data)
                loss = reconstruction_loss_function(output, data)

                print(f"Epoch #{epoch}, Global Step #{global_step}, Local Step #{local_step}: Loss = {loss.item()}")
                summary_writer.add_scalar("Loss/Reconstruction Loss", loss, global_step)
                summary_writer.add_scalar("Batch Size", data.shape[0], global_step)
                summary_writer.add_scalar("Epoch", epoch, global_step)

                summary_writer.add_scalar("Source Batch Statistics/Max Value", torch.max(data), global_step)
                summary_writer.add_scalar("Source Batch Statistics/Avg Value", torch.mean(data), global_step)
                summary_writer.add_scalar("Source Batch Statistics/Std Dev", torch.std(data), global_step)

                summary_writer.add_scalar("Output Batch Statistics/Max Value", torch.max(output), global_step)
                summary_writer.add_scalar("Output Batch Statistics/Avg Value", torch.mean(output), global_step)
                summary_writer.add_scalar("Output Batch Statistics/Std Dev", torch.std(output), global_step)

                if global_step % args.image_summary_interval == 0:
                    print("Logging images ...")
                    batch_sources_image_grid = torchvision.utils.make_grid(data)
                    summary_writer.add_image("Recent Batch Sources", batch_sources_image_grid, global_step)
                    batch_reconstructions_image_grid = torchvision.utils.make_grid(output)
                    summary_writer.add_image("Recent Batch Reconstructions", batch_reconstructions_image_grid, global_step)

                loss.backward()
                optimizer.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="Action to perform (values: download, prepare)")
    parser.add_argument("--working_dir", type=Path, help="Working directory")
    parser.add_argument("--dataset", help="The dataset to work on (values: test, flags, coats_of_arms)")
    parser.add_argument("--img_size", type=int, default=128, help="The size (width) of the training images")
    parser.add_argument("--num_epochs", type=int, default=10, help="The number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=100, help="Batch Size")
    parser.add_argument("--image_summary_interval", type=int, default=100, help="Write image summaries everay N steps")
    args = parser.parse_args()

    if args.action == "train":
        train(args)
