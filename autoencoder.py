import argparse
from typing import Tuple
from pathlib import Path
import mwclient
import mwclient.listing
import mwclient.image
import tqdm
import subprocess
from concurrent.futures import ThreadPoolExecutor

import torch
import torchvision

from models import SimpleConvolutionalAE


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
    png_dir = dataset_dir / "png"

    png_dir.mkdir(exist_ok=True)
    svg_files = list(svg_dir.glob("*.svg"))
    for i, svg_path in tqdm.tqdm(enumerate(svg_files), total=len(svg_files)):
        png_path = png_dir / f"{svg_path.stem}.png"
        if not png_path.exists():
            subprocess.run(["inkscape", f"--export-width={args.img_size}", f"--export-filename={png_path}", str(svg_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_and_prepare_image(file_path: Path):
    pass


def train(args: argparse.Namespace):
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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10, num_workers=1, shuffle=True)

    reconstruction_loss_function = torch.nn.L1Loss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for batch_idx, (data, target) in enumerate(train_loader):
        output = model.forward(data)
        loss = reconstruction_loss_function(output, data)
        print(f"Batch #{batch_idx}: Loss = {loss.item()}")

        loss.backward()
        optimizer.step()

        if batch_idx == 100:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="Action to perform (values: download, prepare)")
    parser.add_argument("--working_dir", type=Path, help="Working directory")
    parser.add_argument("--dataset", help="The dataset to work on (values: test, flags, coats_of_arms)")
    parser.add_argument("--img_size", type=int, default=256, help="The size (width) of the training images")
    args = parser.parse_args()

    if args.action == "download":
        download(args)
    elif args.action == "filter":
        filter(args)
    elif args.action == "prepare":
        prepare(args)
    elif args.action == "train":
        train(args)