import argparse
from typing import Tuple
from pathlib import Path
import mwclient
import mwclient.listing
import mwclient.image
import tqdm
from concurrent.futures import ThreadPoolExecutor


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="Action to perform (values: download)")
    parser.add_argument("--working_dir", type=Path, help="Working directory")
    parser.add_argument("--dataset", help="The dataset to work on (values: test, flags, coats_of_arms)")
    args = parser.parse_args()

    if args.action == "download":
        download(args)