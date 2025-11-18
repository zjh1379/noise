from PIL import Image, ImageDraw
import cv2
import numpy as np
import os, sys
import csv
import math
import argparse
import matplotlib.pyplot as plt 
from matplotlib.colors import hsv_to_rgb

Image.MAX_IMAGE_PIXELS = 100000000

# move the sys args into parser
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


roi_files = os.listdir(args.roi_foldername)
img_files = os.listdir(args.img_foldername)

starts = {}
def slide(img, seg, bbs, folder, idx, size):
    for y0 in range(0, img.size[1], size//2):
        for x0 in range(0, img.size[0], size//2):
            x1, y1 = x0+size, y0+size
            if x1 >= img.size[0]:
                x0, x1 = img.size[0]-size, img.size[0]-1
            if y1 >= img.size[1]:
                y0, y1 = img.size[1]-size, img.size[1]-1
            
            cur_bbs = []
            for bb in bbs:
                
                bb = np.array(bb).reshape((-1,2)) #reshape to shape [-1,2]
                
                # Check that part of the contour is within the patch
                if ( (bb[:,0] >= x0) & (bb[:,0] < x1) & (bb[:,1] >= y0) & (bb[:,1] < y1) ).any():

                    
                    #adjust coordinate origin
                    bb[:,0] = bb[:,0]-x0
                    bb[:,1] = bb[:,1]-y0
                    #drop coordinates outside of bounding box
                    drop_ind = bb[:,0]<0
                    drop_ind = np.logical_or(drop_ind, bb[:,1]<0)
                    drop_ind = np.logical_or(drop_ind, bb[:,0]>size)
                    drop_ind = np.logical_or(drop_ind, bb[:,1]>size)
                    bb = bb[np.logical_not(drop_ind)]
                    
                    #skip any contours which are not polygons (fewer than 3 vertices)
                    if bb.shape[0] < 3:
                        continue
                    
                    #scale to range [0,1]
                    bb = bb/[size,size]
                    #append to cur_bbs
                    cur_bbs.append( [0] + bb.flatten().tolist() )
                    
            
            # skipping when no annotations are available.
            if len(cur_bbs) == 0:
                continue

            # save crops
            img_crop = img.crop((x0,y0,x1,y1))
            yc=str(math.ceil(y0/(size//2)))
            xc=str(math.ceil(x0/(size//2)))
            img_crop.save("{f}/images/img_{id}_{yc}_{xc}.png".format(f=folder, id=idx, yc=yc, xc=xc))

            with open("{f}/labels/img_{id}_{yc}_{xc}.txt".format(f=folder, id=idx, yc=yc, xc=xc), 'w', newline='') as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(cur_bbs)


# check if out_dir exists and create if it doesn't
if not os.path.exists( os.path.join(args.out_foldername,"images") ):
    os.makedirs( os.path.join(args.out_foldername,"images") )
if not os.path.exists( os.path.join(args.out_foldername,"labels") ):
    os.makedirs( os.path.join(args.out_foldername,"labels") )

idx = 0
for roi_filename in roi_files:
    matched = False
    for img_filename in img_files:
        # find corresponding file for ROI-Image
        roi_suffix = roi_filename.split(".")[0].split("_")[-1]
        img_base = img_filename.split(".")[0].strip()
        img_parts = img_base.split("_")

        candidates = []
        if len(img_parts) >= 2:
            candidates.append(img_parts[1].strip())
        candidates.append(img_base)

        found_match = False
        for candidate in candidates:
            candidate = candidate.strip()
            if candidate[: len(roi_suffix)] == roi_suffix:
                found_match = True
                break

        print(roi_suffix)
        print(candidates[0] if candidates else "")

        if not found_match:
            continue

        # print roi and img filenames
        print(roi_filename, " -> ", img_filename)

        # create an instance mask for the entire image
        img_file = os.path.join(args.img_foldername, img_filename)
        csv_file = os.path.join(args.roi_foldername, roi_filename)
        bbs = []
        polygons = []

        with open(csv_file, newline='') as csvfile:
            bbf = csv.reader(csvfile, delimiter=',')

            for row in bbf:
                if not row or len(row) < 6:
                    continue

                header_token = row[0].strip().lower()
                if header_token == "roi_index" or row[1] == "bb_x":
                    continue

                coords = []
                for coor in row[5:]:
                    coor = coor.strip()
                    if coor == "":
                        continue
                    coords.append(float(coor))

                if len(coords) < 6:
                    continue

                polygons.append(coords)
                bbs.append([int(round(coor)) for coor in coords])

        if not polygons:
            print(f"Warning: no annotations in {roi_filename}")
            matched = True
            break

        image_stem = img_filename.split(".")[0]

        if args.skip_sliding:
            with Image.open(img_file) as img:
                width, height = img.size
                label_rows = []

                for coords in polygons:
                    poly = np.array(coords, dtype=np.float32).reshape(-1, 2)
                    if poly.shape[0] < 3:
                        continue

                    poly[:, 0] = np.clip(poly[:, 0] / float(width), 0.0, 1.0)
                    poly[:, 1] = np.clip(poly[:, 1] / float(height), 0.0, 1.0)

                    label_rows.append([0] + poly.flatten().tolist())

                if not label_rows:
                    print(f"Warning: no valid polygons in {roi_filename}")
                    matched = True
                    break

                img.save(
                    os.path.join(
                        args.out_foldername,
                        "images",
                        f"{image_stem}.png",
                    ),
                    format="PNG",
                )

            with open(
                os.path.join(
                    args.out_foldername,
                    "labels",
                    f"{image_stem}.txt",
                ),
                "w",
                newline='',
            ) as f:
                writer = csv.writer(f, delimiter=' ')
                writer.writerows(label_rows)
        else:
            img = Image.open(img_file)
            slide(img, None, bbs, args.out_foldername, idx, args.size)
            img.close()

        idx += 1
        matched = True
        break

    if not matched:
        print(f"Warning: no matching image for {roi_filename}")
