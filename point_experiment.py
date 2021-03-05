import argparse
from pathlib import Path
import os

from numpy.random import randint
from PIL import Image, ImageDraw
from PIL.ImageDraw import ImageDraw
from tqdm import tqdm


def generate_data(target_dir, num_examples, img_w=100, img_h=100, radius=1, num_points=1, point_color="white"):
  target_dir = target_dir / "dummy"
  if not os.path.exists(target_dir):
    os.makedirs(target_dir)

  for i in tqdm(range(num_examples)):
    img = Image.new("RGB", (img_w, img_h), color="black")

    x,y = randint(0, img_w-1), randint(0, img_h-1)

    if radius == 1:
      img.putpixel((x,y), (255, 255, 255))
    else:
      draw = ImageDraw(img)
      ul = (max(0, x-radius), max(0, y-radius))
      lr = (min(img_w, x+radius), min(img_h, y+radius))
      draw.rectangle((ul, lr), fill=(255, 255, 255), outline=(255, 255, 255))

    filename = target_dir / f"example_{i}.png"
    img.save(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", type=Path, help="Working directory")
    parser.add_argument("--img_size", type=int, default=128, help="The size (width) of the training images")
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--radius", type=int, default=8)
    args = parser.parse_args()

    data_dir = args.working_dir / f"dataset_points_{ args.num_images}"
    generate_data(data_dir, args.num_images, img_h=args.img_size, img_w=args.img_size, radius=args.radius)
