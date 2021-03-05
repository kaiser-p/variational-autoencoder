import argparse
import numpy as np
from pathlib import Path
import random
import tqdm
import scipy
from PIL import Image

def generate(args: argparse.Namespace):
  dataset_dir = args.working_dir / "colors" / "dummy_label"
  dataset_dir.mkdir(parents=True, exist_ok=True)

  for i in tqdm.tqdm(range(args.num_images), total=args.num_images):
    img = np.zeros((args.img_size, args.img_size, 3), dtype=np.uint8)
    for j in range(3):
      img[:, :, j] = random.randint(0, 255)
    file_path = dataset_dir / f"img_{i:06d}.png"

    pil_img = Image.fromarray(img, mode="RGB")
    if not file_path.exists():
      pil_img.save(file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", help="Action to perform (values: download, prepare)")
    parser.add_argument("--working_dir", type=Path, help="Working directory")
    parser.add_argument("--dataset", help="The dataset to work on (values: test, flags, coats_of_arms)")
    parser.add_argument("--img_size", type=int, default=100, help="The size (width) of the training images")
    parser.add_argument("--num_images", type=int, default=100000, help="The size (width) of the training images")
    args = parser.parse_args()

    if args.action == "generate":
        generate(args)
