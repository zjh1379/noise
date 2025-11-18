import argparse
import csv
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = 100000000

ALLOWED_IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}
TOKEN_PATTERN = re.compile(r"[0-9a-zA-Z]+")
SAFE_NAME_PATTERN = re.compile(r"[^0-9a-zA-Z_-]+")


def unique(seq):
    seen = set()
    result = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def collect_files(root: Path, extensions=None):
    if not root.exists():
        raise FileNotFoundError(f"Path not found: {root}")

    if root.is_file():
        files = [root]
    else:
        files = [p for p in root.rglob("*") if p.is_file()]

    if extensions:
        exts = {ext.lower() for ext in extensions}
        files = [p for p in files if p.suffix.lower() in exts]

    return sorted(files)


def image_keys(name: str):
    base = Path(name).stem.strip()
    keys = [base.lower()]
    digits = "".join(ch for ch in base if ch.isdigit())
    if digits:
        keys.append(str(int(digits)))
    return unique(keys)


def roi_keys(path: Path):
    base = path.stem.strip()
    keys = [base.lower()]

    if base.lower().startswith("roi_"):
        trimmed = base[4:].strip()
        if trimmed:
            keys.append(trimmed.lower())
            digits_trimmed = "".join(ch for ch in trimmed if ch.isdigit())
            if digits_trimmed:
                keys.append(str(int(digits_trimmed)))

    digits = "".join(ch for ch in base if ch.isdigit())
    if digits:
        keys.append(str(int(digits)))

    return unique(keys)


def extract_tokens(path: Path):
    tokens = set()
    for part in path.parts:
        tokens.update(TOKEN_PATTERN.findall(part.lower()))
    return tokens


def build_image_index(image_files):
    index = defaultdict(list)
    for image_path in image_files:
        tokens = extract_tokens(image_path)
        for key in image_keys(image_path.name):
            index[key].append((image_path, tokens))
    return index


def match_image(roi_path: Path, image_index):
    candidate_map = {}
    for key in roi_keys(roi_path):
        for image_path, tokens in image_index.get(key, []):
            candidate_map[image_path] = tokens

    if not candidate_map:
        return None

    if len(candidate_map) == 1:
        return next(iter(candidate_map))

    roi_tokens = extract_tokens(roi_path)
    best_path = None
    best_score = -1
    for path, tokens in candidate_map.items():
        score = len(tokens & roi_tokens)
        if score > best_score:
            best_path = path
            best_score = score

    if best_path is None:
        best_path = next(iter(candidate_map))

    if best_score == 0:
        print(
            f"Warning: ambiguous match for {roi_path.name}; using {best_path.name}",
            file=sys.stderr,
        )

    return best_path


def parse_roi_polygons(roi_path: Path):
    polygons = []
    with roi_path.open(newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if not row:
                continue
            if row[0].strip().lower() == "roi_index" or (len(row) > 1 and row[1] == "bb_x"):
                continue

            coords = []
            for value in row[5:]:
                value = value.strip()
                if not value:
                    continue
                coords.append(float(value))

            if len(coords) < 6:
                continue

            polygon = np.array(coords, dtype=np.float32).reshape(-1, 2)
            polygons.append(polygon)

    return polygons


def sanitize_name(name: str):
    sanitized = SAFE_NAME_PATTERN.sub("_", name.strip())
    return sanitized or "image"


def ensure_output_dirs(base_dir: Path):
    images_dir = base_dir / "images"
    labels_dir = base_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def write_label_rows(label_path: Path, rows):
    with label_path.open("w", newline="") as label_file:
        for row in rows:
            label_file.write(" ".join(str(value) for value in row))
            label_file.write("\n")


def slide(img, seg, bbs, folder, idx, size):
    folder = Path(folder)
    images_dir = folder / "images"
    labels_dir = folder / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for y0 in range(0, img.size[1], size // 2):
        for x0 in range(0, img.size[0], size // 2):
            x1, y1 = x0 + size, y0 + size
            if x1 >= img.size[0]:
                x0, x1 = img.size[0] - size, img.size[0] - 1
            if y1 >= img.size[1]:
                y0, y1 = img.size[1] - size, img.size[1] - 1

            cur_bbs = []
            for bb in bbs:
                bb = np.array(bb).reshape((-1, 2))

                if (
                    (bb[:, 0] >= x0)
                    & (bb[:, 0] < x1)
                    & (bb[:, 1] >= y0)
                    & (bb[:, 1] < y1)
                ).any():
                    bb[:, 0] = bb[:, 0] - x0
                    bb[:, 1] = bb[:, 1] - y0

                    drop_ind = bb[:, 0] < 0
                    drop_ind = np.logical_or(drop_ind, bb[:, 1] < 0)
                    drop_ind = np.logical_or(drop_ind, bb[:, 0] > size)
                    drop_ind = np.logical_or(drop_ind, bb[:, 1] > size)
                    bb = bb[np.logical_not(drop_ind)]

                    if bb.shape[0] < 3:
                        continue

                    bb = bb / [size, size]
                    cur_bbs.append([0] + bb.flatten().tolist())

            if len(cur_bbs) == 0:
                continue

            img_crop = img.crop((x0, y0, x1, y1))
            yc = str(math.ceil(y0 / (size // 2)))
            xc = str(math.ceil(x0 / (size // 2)))
            crop_name = f"img_{idx}_{yc}_{xc}"
            img_crop.save(images_dir / f"{crop_name}.png")

            write_label_rows(labels_dir / f"{crop_name}.txt", cur_bbs)


def process_with_sliding(roi_files, image_index, out_dir: Path, patch_size: int):
    processed = 0

    for roi_path in roi_files:
        image_path = match_image(roi_path, image_index)
        if image_path is None:
            print(f"Warning: no matching image for {roi_path.name}", file=sys.stderr)
            continue

        polygons = parse_roi_polygons(roi_path)
        if not polygons:
            print(f"Warning: no annotations in {roi_path.name}", file=sys.stderr)
            continue

        bbs = []
        for polygon in polygons:
            if polygon.shape[0] < 3:
                continue
            rounded = np.rint(polygon).astype(int)
            bbs.append(rounded.flatten().tolist())

        if not bbs:
            print(f"Warning: no valid polygons in {roi_path.name}", file=sys.stderr)
            continue

        with Image.open(image_path) as img:
            slide(img, None, bbs, out_dir, processed, patch_size)

        processed += 1

    return processed


def process_full_images(roi_files, image_index, images_dir: Path, labels_dir: Path):
    processed = 0

    for roi_path in roi_files:
        image_path = match_image(roi_path, image_index)
        if image_path is None:
            print(f"Warning: no matching image for {roi_path.name}", file=sys.stderr)
            continue

        polygons = parse_roi_polygons(roi_path)
        if not polygons:
            print(f"Warning: no annotations in {roi_path.name}", file=sys.stderr)
            continue

        with Image.open(image_path) as img:
            width, height = img.size
            label_rows = []

            for polygon in polygons:
                if polygon.shape[0] < 3:
                    continue

                poly = polygon.astype(np.float32).copy()
                poly[:, 0] = np.clip(poly[:, 0] / float(width), 0.0, 1.0)
                poly[:, 1] = np.clip(poly[:, 1] / float(height), 0.0, 1.0)

                label_rows.append([0] + poly.reshape(-1).tolist())

            if not label_rows:
                print(f"Warning: no valid polygons in {roi_path.name}", file=sys.stderr)
                continue

            image_stem = image_path.stem

            output_image_path = images_dir / f"{image_stem}.png"
            img.save(output_image_path, format="PNG")

        output_label_path = labels_dir / f"{image_stem}.txt"
        write_label_rows(output_label_path, label_rows)

        processed += 1

    return processed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi_foldername", type=str, default="roi")
    parser.add_argument("--img_foldername", type=str, default="img")
    parser.add_argument("--out_foldername", type=str, default="bbs")
    parser.add_argument("--size", type=int, default=832)
    parser.add_argument(
        "--skip_sliding",
        action="store_true",
        help="Assume images are already patches and skip sliding window cropping.",
    )

    args = parser.parse_args()

    roi_root = Path(args.roi_foldername)
    img_root = Path(args.img_foldername)
    out_root = Path(args.out_foldername)

    try:
        roi_files = collect_files(roi_root, {".csv"})
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if not roi_files:
        print(f"No ROI CSV files found in {roi_root}", file=sys.stderr)
        sys.exit(1)

    try:
        image_files = collect_files(img_root, ALLOWED_IMAGE_EXTS)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if not image_files:
        print(f"No image files found in {img_root}", file=sys.stderr)
        sys.exit(1)

    image_index = build_image_index(image_files)

    images_dir, labels_dir = ensure_output_dirs(out_root)

    if args.skip_sliding:
        processed = process_full_images(roi_files, image_index, images_dir, labels_dir)
    else:
        processed = process_with_sliding(roi_files, image_index, out_root, args.size)

    print(f"Processed {processed} ROI file(s).")


if __name__ == "__main__":
    main()
