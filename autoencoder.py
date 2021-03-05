from __future__ import annotations

import argparse
from typing import Tuple
from pathlib import Path
import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision
import torch.utils.tensorboard
import torch.utils.data
from models import SimpleConvolutionalAE

try:
    import mwclient
    import mwclient.listing
    import mwclient.image
except ImportError:
    pass

def download_image(image: mwclient.image.Image, target_path: Path):
    if target_path.exists():
        return
    with target_path.open("wb") as f_img:
        image.download(f_img)


def download(args: argparse.Namespace):
    def removeprefix(string, prefix):
        if string.startswith(prefix):
            return string[len(prefix):]
        return string

    category_name_map = {
        "test": "SVG_military_flags_of_Vatican_City",
        "flags": "SVG_flags",
        "coats_of_arms": "SVG_coats_of_arms"
    }
    dataset_dir = args.working_dir / f"dataset_{args.dataset}"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    svg_target_dir = args.working_dir / f"dataset_{args.dataset}" / "svg"
    svg_target_dir.mkdir(parents=True, exist_ok=True)

    site = mwclient.Site("commons.wikimedia.org")
    futures = []
    with ThreadPoolExecutor(max_workers=20) as ex:
        processed_categories = set()
        processed_images = set()
        categories_to_process = [site.Categories[category_name_map[args.dataset]]]
        categories_discovered = []
        images_discovered = []
        while categories_to_process:
            discovered_category = categories_to_process.pop()
            categories_discovered.append(discovered_category)
            print(f"{len(categories_discovered)} categories processed, {len(images_discovered)} images found, {len(categories_to_process)} categories left, processing category: {discovered_category.name}")
            for element in discovered_category:
                if isinstance(element, mwclient.listing.Category):
                    if element.name not in processed_categories:
                        processed_categories.add(element.name)
                        categories_to_process.append(element)
                elif isinstance(element, mwclient.image.Image):
                    if element.name not in processed_images:
                        processed_images.add(element.name)
                        images_discovered.append(element)
                        target_dir = svg_target_dir / removeprefix(discovered_category.name, "Category:").replace(" ", "_")
                        target_dir.mkdir(parents=True, exist_ok=True)
                        target_path = target_dir / removeprefix(element.name, "File:").replace(" ", "_")
                        futures.append(ex.submit(download_image, image=element, target_path=target_path))
        print(f"{len(categories_discovered)} categories discovered")
        print(f"{len(images_discovered)} images discovered")


def filter(args: argparse.Namespace):
    filter_patterns = ["flag-map", "map-flag", "flag_map", "map_flag", "silhouette", "nuvola", "icon"]

    dataset_dir = args.working_dir / f"dataset_{args.dataset}"
    svg_dir = dataset_dir / "svg"
    filtered_dir = dataset_dir / "filtered"

    filtered_dir.mkdir(exist_ok=True)

    for path in svg_dir.rglob("*"):
        if path.is_dir():
            continue
        if not path.name.lower().endswith(".svg") or any(fp in path.name.lower() for fp in filter_patterns):
            path.rename(filtered_dir / path.name)
            print(f"Filtered: {path}")
        else:
            path.rename(svg_dir / path.name)


def prepare(args: argparse.Namespace):
    dataset_dir = args.working_dir / f"dataset_{args.dataset}"
    svg_dir = dataset_dir / "svg"
    png_dir = dataset_dir / "png" / "dummy_label"

    png_dir.mkdir(exist_ok=True)
    svg_files = list(svg_dir.glob("*.svg"))
    for i, svg_path in tqdm.tqdm(enumerate(svg_files), total=len(svg_files)):
        png_path = png_dir / f"{svg_path.stem}.png"
        if not png_path.exists():
            subprocess.run(["inkscape", f"--export-width={args.img_size}", f"--export-filename={png_path}", str(svg_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def train(args: argparse.Namespace):
    train_dir = args.working_dir / "train"
    logs_dir = args.working_dir / "train" / "summary"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleConvolutionalAE(img_dim=args.img_size, bottleneck_dim=32)
    model.to(device)
    print(model)

    image_transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.img_size, args.img_size)),  # TODO: Only resize the canvas
        torchvision.transforms.ToTensor()
    ])

    train_data_dir = args.working_dir / f"dataset_{args.dataset}" / "png"
    train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=image_transforms)

    reconstruction_loss_function = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-8)

    with torch.utils.tensorboard.SummaryWriter(log_dir=logs_dir) as summary_writer:
        global_step = 0
        for epoch in range(args.num_epochs):
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=50, num_workers=1, shuffle=True)
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

                if global_step % 10 == 0:
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
    parser.add_argument("--img_size", type=int, default=256, help="The size (width) of the training images")
    parser.add_argument("--num_epochs", type=int, default=10, help="The number of epochs to train")
    args = parser.parse_args()

    if args.action == "download":
        download(args)
    elif args.action == "filter":
        filter(args)
    elif args.action == "prepare":
        prepare(args)
    elif args.action == "train":
        train(args)