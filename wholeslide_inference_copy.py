import os, sys
import glob
from ultralytics import YOLO
import csv
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from random import randint
import torch
import torchvision
import argparse
import math
import numpy as np
import json
from skspatial.measurement import area_signed

# um/pixel length
UM_PER_PIXEL = 0.7784
UM_PER_PATCH = UM_PER_PIXEL * 832
#0.5945 µm2/pixel
MAX_DET = 30000
Image.MAX_IMAGE_PIXELS = 1000000000

DEVICE = torch.device('cuda:0')

# Flags for duplicate handling (kept global for minimal code changes)
# - USE_IOU_LOCAL_NMS: 0 to use original coverage-based local NMS; 1 to use IoU-based local NMS
# - LOCAL_NMS_IOU: IoU threshold for local NMS when IoU-based is enabled
# - ENABLE_GLOBAL_NMS: 1 to run a global NMS across all detections after neighborhood merging
# - GLOBAL_NMS_IOU: IoU threshold for the global NMS
USE_IOU_LOCAL_NMS = 0
LOCAL_NMS_IOU = 0.6
ENABLE_GLOBAL_NMS = 0
GLOBAL_NMS_IOU = 0.5


def scale_boxes(boxes, num_images, img_ind, img_scale):
    boxes[:,(0,2)] = boxes[:,(0,2)] + (img_scale[0]/2)*img_ind[0] # x
    boxes[:,(1,3)] = boxes[:,(1,3)] + (img_scale[1]/2)*img_ind[1] # y
    return boxes
    
def scale_masks(masks, num_images, img_ind, img_scale):
    for m in range(len(masks)):
        masks[m] = masks[m] + (img_scale/2)*img_ind # x
    return masks

# Credit to torchvision/ops/boxes.py
def box_inter_union(boxes1, boxes2):
    area1 = torchvision.ops.box_area(boxes1)
    area2 = torchvision.ops.box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union
    
def local_nms(box_results, mask_results, img_size): # Non-maximum suppression for patch results

    if box_results[1][1].numel() == 0:
        return torch.tensor([], device=DEVICE), []
    
    keep_bool_master = []
    tmp_boxes = []

    tmp_boxes.append(box_results[0][0])
    tmp_boxes.append(box_results[0][1])
    tmp_boxes.append(box_results[0][2])
    tmp_boxes.append(box_results[1][0])

    tmp_boxes.append(box_results[1][2])
    tmp_boxes.append(box_results[2][0])
    tmp_boxes.append(box_results[2][1])
    tmp_boxes.append(box_results[2][2])
    tmp_boxes_torch = torch.cat(tmp_boxes)
    
    # If no neighboring predictions, skip
    if torch.numel(tmp_boxes_torch)==0:
        return torch.tensor([], device=DEVICE), []
    
    for box in box_results[1][1]:
        intersection = box_inter_union(box[:4].unsqueeze(0), tmp_boxes_torch[:,:4])[0]

        surf_area = torchvision.ops.box_area( box[:4].unsqueeze(0) )[0]
        comp = ((surf_area - intersection)/surf_area) < 0.1
        if torch.any( comp ):
            area = torchvision.ops.box_area( tmp_boxes_torch[comp[0],:4] )
            # If boxes overlap significantly, keep larger box
            if (torch.all(surf_area > area) \
                and (box[0] < img_size[0]) \
                and (box[1] < img_size[1])):
                # Also checks that the box starts inside the original image
                keep_bool_master.append( True )
            else:
                keep_bool_master.append( False )
        else:
            keep_bool_master.append( True )
        
        
    if not any(keep_bool_master):
        return torch.tensor([], device=DEVICE), []
    
    box_results = box_results[1][1][keep_bool_master]
    mask_results = [ mask_results[1][1][i] for i in range(len(keep_bool_master)) if keep_bool_master[i] ]
    return box_results, mask_results


def local_nms_iou(box_results, mask_results, img_size, iou_thresh=0.6):
    """IoU-based local NMS over a 3x3 neighborhood.
    - Keeps a detection in the center cell only if no neighbor overlaps it with IoU >= iou_thresh
      and at least the same confidence score.
    - This symmetric IoU criterion is more robust than coverage ratio to small coordinate jitters.
    """
    if box_results[1][1].numel() == 0:
        return torch.tensor([], device=DEVICE), []

    keep_bool_master = []
    tmp_boxes = []

    tmp_boxes.append(box_results[0][0])
    tmp_boxes.append(box_results[0][1])
    tmp_boxes.append(box_results[0][2])
    tmp_boxes.append(box_results[1][0])
    tmp_boxes.append(box_results[1][2])
    tmp_boxes.append(box_results[2][0])
    tmp_boxes.append(box_results[2][1])
    tmp_boxes.append(box_results[2][2])
    tmp_boxes_torch = torch.cat(tmp_boxes)

    if torch.numel(tmp_boxes_torch)==0:
        return torch.tensor([], device=DEVICE), []

    for box in box_results[1][1]:
        inter, union = box_inter_union(box[:4].unsqueeze(0), tmp_boxes_torch[:,:4])
        iou = inter / (union + 1e-9)
        conflict = (iou[0] >= iou_thresh)
        if torch.any(conflict):
            neighbor_scores = tmp_boxes_torch[:,4]
            keep = torch.all(neighbor_scores[conflict] < box[4])
            keep = bool(keep and (box[0] < img_size[0]) and (box[1] < img_size[1]))
            keep_bool_master.append( keep )
        else:
            keep_bool_master.append( True )

    if not any(keep_bool_master):
        return torch.tensor([], device=DEVICE), []

    box_results = box_results[1][1][keep_bool_master]
    mask_results = [ mask_results[1][1][i] for i in range(len(keep_bool_master)) if keep_bool_master[i] ]
    return box_results, mask_results


def global_nms(boxes, masks, iou_thresh=0.5):
    """Global box-level NMS across all detections after neighborhood merging.
    - Uses torchvision.ops.nms on [x1,y1,x2,y2] with scores at index 4.
    - Returns filtered boxes and corresponding masks subset.
    """
    if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
        return boxes, masks
    keep_idx = torchvision.ops.nms(boxes[:,:4], boxes[:,4], iou_thresh)
    boxes = boxes[keep_idx]
    masks = [masks[i] for i in keep_idx.tolist()]
    return boxes, masks

def flatten_detection_grid(box_grid, mask_grid):
    """Flatten 2D patch grids into a single tensor/list pair."""
    boxes = []
    masks = []
    for r in range(1, len(box_grid) - 1):
        for c in range(1, len(box_grid[r]) - 1):
            cell_boxes = box_grid[r][c]
            if torch.is_tensor(cell_boxes) and cell_boxes.numel() > 0:
                boxes.append(cell_boxes)
                masks.extend(mask_grid[r][c])
    if boxes:
        return torch.cat(boxes), masks
    return torch.tensor([], device=DEVICE), []

def save_process_stage(img, boxes, masks, patches, save_path, stage_label):
    """Save a visualization of detections for a specific processing stage."""
    if not isinstance(img, Image.Image):
        return
    canvas = img.copy()
    draw = ImageDraw.Draw(canvas, 'RGBA')
    font = ImageFont.load_default()
    label_text = stage_label if stage_label else ""
    if label_text:
        draw.text((10, 10), label_text, font=font, fill="yellow")

    if isinstance(boxes, torch.Tensor) and boxes.numel() > 0:
        boxes_cpu = boxes.detach().cpu()
        for i, box in enumerate(boxes_cpu):
            coords = box[:4].int().tolist()
            x1 = max(0, coords[0])
            y1 = max(0, coords[1])
            x2 = min(canvas.size[0] - 1, coords[2])
            y2 = min(canvas.size[1] - 1, coords[3])
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
            draw.text((x1, y1), f"{i+1}", font=font, fill="red")

            if i < len(masks):
                mask = masks[i]
                if mask is not None:
                    polygon = np.array(mask).astype(int).flatten().tolist()
                    if len(polygon) >= 6:
                        color = (randint(0,255), randint(0,255), randint(0,255))
                        draw.polygon(polygon, fill=color+(100,), outline="blue")

    if patches:
        for patch in patches:
            draw.rectangle([(patch[0], patch[1]), (patch[2], patch[3])], outline="green", width=1)

    canvas.save(save_path)

def save_patch_predictions(patch_img, results, patch_coords, save_dir, img_root_name, patch_index):
    """Save each patch crop with raw model predictions."""
    if not isinstance(patch_img, Image.Image) or not results:
        return
    os.makedirs(save_dir, exist_ok=True)
    canvas = patch_img.copy()
    draw = ImageDraw.Draw(canvas, 'RGBA')
    font = ImageFont.load_default()
    draw.text((10, 10), f"Patch {patch_index} @ ({patch_coords[0]}, {patch_coords[1]})", font=font, fill="yellow")
    det_counter = 0
    for det in results:
        if det.boxes is None or det.boxes.xyxy is None:
            continue
        boxes_xyxy = det.boxes.xyxy.detach().cpu().numpy()
        scores = det.boxes.conf.detach().cpu().numpy() if det.boxes.conf is not None else None
        masks = det.masks.xy if det.masks is not None and det.masks.xy is not None else []
        for idx, box in enumerate(boxes_xyxy):
            det_counter += 1
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=2)
            label = f"{det_counter}"
            if scores is not None and idx < len(scores):
                label = f"{label}:{scores[idx]:.2f}"
            draw.text((x1, y1), label, font=font, fill="red")
            if idx < len(masks):
                polygon = np.array(masks[idx]).astype(float).flatten().tolist()
                if len(polygon) >= 6:
                    color = (randint(0,255), randint(0,255), randint(0,255))
                    draw.polygon(polygon, fill=color+(80,), outline="blue")
    filename = f"{img_root_name}_patch_{patch_index:05d}_x{patch_coords[0]}_y{patch_coords[1]}.png"
    canvas.save(os.path.join(save_dir, filename))

def save_individual_masks(img, boxes, masks, base_dir, img_root_name, stage_tag):
    """Save an image per mask showing local crop and polygon."""
    if not isinstance(img, Image.Image):
        return
    if not isinstance(boxes, torch.Tensor) or boxes.numel() == 0:
        return
    os.makedirs(base_dir, exist_ok=True)
    stage_dir = os.path.join(base_dir, stage_tag)
    os.makedirs(stage_dir, exist_ok=True)
    boxes_cpu = boxes.detach().cpu()
    width, height = img.size
    for idx, box in enumerate(boxes_cpu):
        coords = box[:4].tolist()
        if len(coords) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in coords]
        margin = 15
        crop_box = (
            max(0, x1 - margin),
            max(0, y1 - margin),
            min(width, x2 + margin),
            min(height, y2 + margin)
        )
        if crop_box[0] >= crop_box[2] or crop_box[1] >= crop_box[3]:
            continue
        crop = img.crop(crop_box)
        draw = ImageDraw.Draw(crop, 'RGBA')
        font = ImageFont.load_default()
        draw.rectangle(
            [(x1 - crop_box[0], y1 - crop_box[1]), (x2 - crop_box[0], y2 - crop_box[1])],
            outline="red",
            width=3
        )
        score = float(box[4].item()) if box.numel() > 4 else None
        label_txt = f"{stage_tag} #{idx}"
        if score is not None:
            label_txt = f"{label_txt} ({score:.2f})"
        draw.text((5, 5), label_txt, font=font, fill="yellow")
        if idx < len(masks):
            mask = masks[idx]
            if mask is not None:
                mask_arr = np.array(mask, dtype=float)
                if mask_arr.size >= 6:
                    mask_arr[:,0] -= crop_box[0]
                    mask_arr[:,1] -= crop_box[1]
                    polygon = mask_arr.flatten().tolist()
                    color = (randint(0,255), randint(0,255), randint(0,255))
                    draw.polygon(polygon, fill=color+(80,), outline="blue")
        filename = f"{img_root_name}_{stage_tag}_mask_{idx:05d}.png"
        crop.save(os.path.join(stage_dir, filename))
    
def inference(model, img, img_filename, size, out_dir, save_process_viz=False):
    
    empty_tensor = torch.tensor([], device=DEVICE)
    
    # divide size of image by size of patch/2
    num_patches = ( np.array(img.size) / (size/2) ).astype(int)
    
    # Run inference on each image
    box_results = []
    mask_results = []

    
    box_results.append( [empty_tensor for _ in range(0, img.size[0], size//2)] )
    box_results[-1] += [empty_tensor, empty_tensor]
    mask_results.append( [[] for _ in range(0, img.size[0], size//2)] )
    mask_results[-1] += [[], []]
    
    patches = [] # For debugging
    process_stage_dir = None
    patch_viz_dir = None
    mask_viz_dir = None
    img_root_name = os.path.splitext(img_filename)[0]
    if save_process_viz:
        process_root_dir = os.path.join(out_dir, "process_viz")
        process_img_dir = os.path.join(process_root_dir, img_root_name)
        process_stage_dir = os.path.join(process_img_dir, "stages")
        patch_viz_dir = os.path.join(process_img_dir, "patches")
        mask_viz_dir = os.path.join(process_img_dir, "individual_masks")
        os.makedirs(process_stage_dir, exist_ok=True)
        os.makedirs(patch_viz_dir, exist_ok=True)
        os.makedirs(mask_viz_dir, exist_ok=True)
    
    patch_counter = 0
    for y0 in range(0, img.size[1], size//2):
        box_results.append([ empty_tensor ])
        mask_results.append([ [] ])
        for x0 in range(0, img.size[0], size//2):
            
            x1, y1 = x0+size, y0+size
            
            patches += [[x0, y0, x1, y1]] # For visualizing crop grid later
            
            # Create crops (pasting onto blank white image, since the default
            # PIL crop function fills with black, causing false detections):
            img_crop = Image.new('RGB', (size, size), (255, 255, 255))
            img_crop.paste(img, (-x0, -y0))
            
            # # Old cropping function (default black fill caused bad detections):
            # img_crop = img.crop((x0,y0,x1,y1))
            
            # # Save cropped images for debugging:
            # img_crop.save( "{f}/{a}_{b}_{id}".format(f=out_dir, a = str(x0), b = str(y0), id=img_filename) )
            
            yc=math.ceil(y0/(size//2))
            xc=math.ceil(x0/(size//2))
            
            results = model( img_crop, verbose=False, device=DEVICE )
            img_ind = np.array((xc,yc))
            patch_counter += 1
            if save_process_viz:
                save_patch_predictions(
                    img_crop,
                    results,
                    (x0, y0, x1, y1),
                    patch_viz_dir,
                    img_root_name,
                    patch_counter
                )
            
            # Scale the predictions back to their proper size
            for r in range(len(results)):
                boxes = results[r].boxes.data.clone()
                
                if boxes.numel() != 0: # if osteoclasts detected
                    boxes = scale_boxes(boxes, num_patches, img_ind, (size,size))
                    masks = scale_masks(results[r].masks.xy, num_patches, img_ind, np.array((size,size)))
                    box_results[-1].append( boxes )
                    mask_results[-1].append( masks )
                else:
                    box_results[-1].append( empty_tensor )
                    mask_results[-1].append( [] )
                    
        box_results[-1].append( empty_tensor )
        mask_results[-1].append( [] )
    
    box_results.append( [empty_tensor for _ in range(0, img.size[0], size//2)] )
    box_results[-1] += [empty_tensor, empty_tensor]
    mask_results.append( [[] for _ in range(0, img.size[0], size//2)] )
    mask_results[-1] += [[], []]
    
    objects_found = True if box_results else False
    
    if save_process_viz:
        raw_boxes, raw_masks = flatten_detection_grid(box_results, mask_results)
        raw_stage_path = os.path.join(process_stage_dir, f"{img_root_name}_raw.png")
        save_process_stage(img, raw_boxes, raw_masks, patches, raw_stage_path, "Stage: raw detections")
        if isinstance(raw_boxes, torch.Tensor) and raw_boxes.numel() > 0:
            save_individual_masks(img, raw_boxes, raw_masks, mask_viz_dir, img_root_name, "raw")

    if objects_found:
    
        new_box_results = []
        new_mask_results = []
        for r in range(1,len(box_results)-1):
            for c in range(1,len(box_results[r])-1):
                
                # If empty
                if torch.numel(box_results[r][c])==0:
                    continue
                
                # Choose NMS implementation: original coverage-based or IoU-based
                nms_fn = local_nms_iou if USE_IOU_LOCAL_NMS else local_nms
                output = nms_fn([ b[c-1:c+2] for b in box_results[r-1:r+2] ], [ m[c-1:c+2] for m in mask_results[r-1:r+2] ], img.size)
                
                # print(r, c, output[0], '\n')
                
                new_box_results.append(output[0])
                new_mask_results += output[1]

        # If no osteoclasts are detected in image, this will handle the output
        if len(new_box_results) > 0:
            box_results = torch.cat(new_box_results)
            mask_results = new_mask_results
        else:
            box_results = [0]
            mask_results = new_mask_results

        if save_process_viz and isinstance(box_results, torch.Tensor) and box_results.numel() > 0:
            local_stage_path = os.path.join(process_stage_dir, f"{img_root_name}_local_nms.png")
            save_process_stage(img, box_results, mask_results, patches, local_stage_path, "Stage: after local NMS")
            save_individual_masks(img, box_results, mask_results, mask_viz_dir, img_root_name, "local_nms")

    # Optional global NMS across all boxes
    if isinstance(box_results, torch.Tensor) and box_results.numel() > 0 and ENABLE_GLOBAL_NMS:
        box_results, mask_results = global_nms(box_results, mask_results, iou_thresh=GLOBAL_NMS_IOU)
        if save_process_viz and isinstance(box_results, torch.Tensor) and box_results.numel() > 0:
            global_stage_path = os.path.join(process_stage_dir, f"{img_root_name}_global_nms.png")
            save_process_stage(img, box_results, mask_results, patches, global_stage_path, "Stage: after global NMS")
            save_individual_masks(img, box_results, mask_results, mask_viz_dir, img_root_name, "global_nms")

    
    with open("{f}/{id}".format(f=out_dir, id=img_filename[:-4]+".txt"), 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow( ["box_x1","box_y1","box_x2","box_y2","objectness_score","mask_x1","mask_y1","mask_x2","mask_y2","..."] )
        if (len(box_results)) > 1:
            for i in range(len(box_results)):
                writer.writerow( box_results[i].tolist()[:-1] + mask_results[i].flatten().tolist() )
        else:
             return f.write("No osteoclasts detected")

    # Draw boxes on original image
    img1 = ImageDraw.Draw(img, 'RGBA')
    font = ImageFont.load_default()
    
    for i, box in enumerate(box_results):
        box = box[:4].type(torch.int)
        
        # The min/max modifiers seem to help boxes on the edge show up:
        shape = [(max(0, box[0]), min(box[1], img.size[0]-1)), \
                 (max(0, box[2]), min(box[3], img.size[1]-1))]
                 
        img1.rectangle(shape, outline="red", width=3)
        img1.text((box[0], box[1]), str(i + 2), font = font, fill="red")
        # print(mask_results[i].astype(int).flatten().tolist())
        mask = mask_results[i].astype(int).flatten().tolist()
        if len(mask) >= 6:
            color = (randint(0,255),randint(0,255),randint(0,255))
            img1.polygon(mask, fill=color+(125,), outline="blue")
    
    for i, patch in enumerate(patches):
        img1.rectangle([(patch[0], patch[1]), (patch[2], patch[3])], outline="green", width=1)
            
    img.save( "{f}/{id}".format(f=out_dir, id=img_filename) )
    
    if objects_found:
        return [{"boxes":box_results[:,:4], "scores":box_results[:,4], "labels":box_results[:,5].int()}]
    else:
        return [{"boxes":[], "scores":[], "labels":[]}]

def count_ocls_from_output(img_dir, out_dir):
    
    # This script will count each newline for the files in the output directory

    #This will save the output_files to a list from the output directory and only include the txt files
    output_files = glob.glob((out_dir) + "*.txt")

    #Add the image directory name to output file
    image_dir = (img_dir.rsplit("/")) # split usr give image directory into a list split by /

    # This will make sure that a blank name is not written as part of the ocl_count output folder name
    if len(image_dir[-1]) > 0:
        output_name = image_dir[-1]
    else:
        output_name = image_dir[-2]
    
    col_name = ["Image_Name", "Ocl_Count"] # Add the column names to top of csv.

    csv_file_name = "ocl_counts_" + str(output_name) +".csv" # file name

    with open(csv_file_name, "a", newline = '') as csvfile:
        if os.stat(csv_file_name).st_size == 0: # only put the column name row in when the file is empty (at start)
            writer = csv.writer(csvfile)
            writer.writerow(col_name)
        
    split_dir = len(out_dir)
    #To iterate over each file in that output directory
    for file in output_files:
        counts_list = []
        with open(file, "r") as f: # f is now the object of each file
            as_string = str(f.read())
            split_string = as_string.split("\n")
            counts_list.append("{id}".format(id=f.name[split_dir:-4]))
            counts_list.append(str(len(split_string[1:-1])))
            with open(csv_file_name, "a", newline = '') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(counts_list)
        csvfile.close
    
#Below functions are required for area calculations
def masking_coordinates_to_list(out_dir):

    # This function will count each newline for the files in the output directory

    #This will save the output_files to a list from the output directory and only include the txt files
    output_files = glob.glob((out_dir) + "*.txt")

    dir_list = [] # List will contain each .txt file name containing the masking coordinates. 
    for file in os.listdir(out_dir):
        if file.endswith('.txt'):
            dir_list.append(file)

    length_files_in_dir = (len(dir_list)) # Save how many files are in the directory
    
    counter = 0
    coordinate_dict = {}  # Save each file name as a key to the masking coordinate value
    #To iterate over each file in that output directory
    while len(coordinate_dict) != length_files_in_dir:
        for file in (output_files):
            with open(file, "r") as f: # f is now the object of each file
                as_string = str(f.read())
                split_string = as_string.split("\n") # Split string has each osteoclast masking coordinates in an element of a list.
    
                coordinate_dict[dir_list[counter]] = split_string
                counter += 1
    #print(coordinate_dict)
    return coordinate_dict # The coordinate dict will have each file name as a key and the masking coordinates as a value.
              
def calculate_pixel_area(coordinate_list_as_floats):
    '''This function will create a 2d array of the masking coordinates
    as input for the area_signed function which utilizes the shoelace
    formula to calculate area of an irregular polygon. 

    Returns the pixel area of each osteoclast. '''

    for i in coordinate_list_as_floats:
         if isinstance(i, float):      
            array_2d = np.array(coordinate_list_as_floats).reshape((len(coordinate_list_as_floats))//2,2) #Create numpy array

            pixel_area = (abs(area_signed(array_2d))) # Output Pixel area of each osteoclast using shoelace formula

            return (pixel_area)

def pixel_area_to_um_sqrd(pixel_area, um_per_pixel):
     
    '''This function will calculate the area of each ocl in um^2.
    Calculations will be based on um_per_pixel ratio given by the user.'''

    um_sqrd_area = pixel_area * ((um_per_pixel) ** 2)
    return round(um_sqrd_area, 3)

def total_area_per_well(area_list):

    # Each area of an ocl will be added to the total area 
    total_area_per_well_sum = 0

    for i in area_list:
         total_area_per_well_sum += i

    return round(total_area_per_well_sum, 3)


def percent_ocl_area_per_well(total_area, well_area_in_pixels):

    '''Function will calculate the percent of osteoclast area on each well.
    The well_area_in_pixels is set to the user entered area.'''
    perecent_area = (total_area/well_area_in_pixels)*100

    return round(perecent_area, 3)

def write_area_to_output(img_dir, total_area_per_well_sum, percent_area, out_dir ,key):

    '''Function will write each area to a csv.'''

    image_dir = (img_dir.rsplit("/"))

    if len(image_dir[-1]) > 0:
        output_name = image_dir[-1]
    else:
        output_name = image_dir[-2]
    # Add a row for column names to top of file

    col_name = ["Image_Name", "Total_Area", "%_Area"] # First row of csv with column names

    csv_file_name = "ocl_area_" + str(output_name) + ".csv" # name of file to store data

    with open(csv_file_name, "a", newline = '') as csvfile:
        if os.stat(csv_file_name).st_size == 0: # only put the column name row in when the file is empty (at start)
            writer = csv.writer(csvfile)
            writer.writerow(col_name)

    area_output_tuple = [key[:-4], str(total_area_per_well_sum), str(percent_area)] # tuple to store the row to write to csv
    with open(csv_file_name, "a", newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(area_output_tuple)
    csvfile.close

def write_individual_ocl_area_to_output(img_dir, all_individual_ocl_areas_dict):
    '''Going to output a file with all ocl areas'''

    image_dir = (img_dir.rsplit("/"))

    if len(image_dir[-1]) > 0:
        output_name = image_dir[-1]
    else:
        output_name = image_dir[-2]

    col_name = ["Image_Name"]

    csv_file_name = "ocl_individual_areas_" + str(output_name) + ".csv"

    with open(csv_file_name, "a", newline = '') as csvfile:
        if os.stat(csv_file_name).st_size == 0: # only put the column name row in when the file is empty (at start)
            writer = csv.writer(csvfile)
            writer.writerow(col_name)

    with open (csv_file_name,'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in all_individual_ocl_areas_dict.items():
            writer.writerow([key[:-4], *value]) # write a row as image name, then the individual areas. 

    csvfile.close()


def main(argv):
    
    # move the sys args into parser
    parser = argparse.ArgumentParser()

    #Add option to utilize params.json file
    parser.add_argument("--params", type=str, default=None) #should be the params file path

    parser.add_argument("--img_foldername", type=str, default="img")
    parser.add_argument("--out_foldername", type=str, default="out")
    parser.add_argument("--model_path", type=str, default="out")
    parser.add_argument("--ratio", type=float, default=0.7784) #um per pixel
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--total_well_area_in_pixels", type = int, default = 0)
    # New options for overlap/duplicate handling test
    # The following flags only toggle algorithms; sliding overlap ratio remains unchanged (50%).
    parser.add_argument("--use_iou_local_nms", type=int, default=0, help="1 to enable IoU-based local NMS; 0 to use original coverage-based")
    parser.add_argument("--local_nms_iou", type=float, default=0.6, help="IoU threshold for local IoU-NMS")
    parser.add_argument("--enable_global_nms", type=int, default=0, help="1 to enable a final global NMS")
    parser.add_argument("--global_nms_iou", type=float, default=0.5, help="IoU threshold for global NMS")
    parser.add_argument("--save_process_viz", type=int, default=0, help="1 to export process visualization images")
    

    args = parser.parse_args()

    json_parameter = args.params

    # Initialize duplicate-handling options from CLI defaults first
    use_iou_local_nms = int(args.use_iou_local_nms)
    local_nms_iou = float(args.local_nms_iou)
    enable_global_nms = int(args.enable_global_nms)
    global_nms_iou = float(args.global_nms_iou)
    save_process_viz = int(args.save_process_viz)

    if json_parameter != None: # if params file given
       with open(json_parameter) as params_file: # open the params file
           data = json.load(params_file) # store params as dict in data

    for param,argument in data.items(): # parse params dict, assigning each usr argument to correct variable
        
        if param == "model_path":
            model_path = argument # the model path is equal to json given argument
            
        if param == "img_foldername":
            img_dir = argument

        if param ==  "out_foldername":
            out_dir = argument

        if param == "ratio":
            um_per_pixel = argument
            patch_size = int( UM_PER_PATCH/um_per_pixel )
        if param == "use_iou_local_nms":
            use_iou_local_nms = int(argument)
        if param == "local_nms_iou":
            local_nms_iou = float(argument)
        if param == "enable_global_nms":
            enable_global_nms = int(argument)
        if param == "global_nms_iou":
            global_nms_iou = float(argument)
        if param == "save_process_viz":
            save_process_viz = int(argument)

        if param == "total_well_area_in_pixels":
            well_area_in_pixels = argument 

        if "total_well_area_in_pixels" not in data:
            well_area_in_pixels = 0 # sets the well_area to 0, will return None for % area

        if param == "device":
            usr_device = argument

        if "device" not in data:
            usr_device = "cpu" # sets the usr_device to default cpu if it's not provided 

    if json_parameter == None: # If no json params file provided, will use user arguments to run the command
    
        um_per_pixel = args.ratio
        patch_size = int( UM_PER_PATCH/um_per_pixel )
        
        out_dir = args.out_foldername
        img_dir = args.img_foldername

        well_area_in_pixels = args.total_well_area_in_pixels

        model_path = args.model_path
        model = YOLO(model_path)

        usr_device = args.device

        use_iou_local_nms = int(args.use_iou_local_nms)
        local_nms_iou = float(args.local_nms_iou)
        enable_global_nms = int(args.enable_global_nms)
        global_nms_iou = float(args.global_nms_iou)
        save_process_viz = int(args.save_process_viz)

    global DEVICE
    DEVICE = torch.device(usr_device)
    
    if out_dir == img_dir:
        print("Error: Input directory equals output directory. Please specify a unique output directory.")
        return
    
    # check if out_dir exists and create if it doesn't
    if not os.path.exists( out_dir ):
        os.makedirs( out_dir )
        
    model = YOLO(model_path)
        
    img_files = [ file for file in os.listdir(img_dir) if not file.startswith(".") ]
    for img_filename in img_files:
        print(img_filename)
        
        img = Image.open( os.path.join(img_dir, img_filename) )
        
        # Update global flags from CLI/JSON 
        global USE_IOU_LOCAL_NMS, LOCAL_NMS_IOU, ENABLE_GLOBAL_NMS, GLOBAL_NMS_IOU
        USE_IOU_LOCAL_NMS = use_iou_local_nms
        LOCAL_NMS_IOU = local_nms_iou
        ENABLE_GLOBAL_NMS = enable_global_nms
        GLOBAL_NMS_IOU = global_nms_iou

        pred = inference(model, img, img_filename, patch_size, out_dir, save_process_viz=bool(save_process_viz))

    
    count_ocls_from_output(img_dir, out_dir)

    split_string = masking_coordinates_to_list(out_dir) # Split string variable is now a dictionary, where each each key is a txt and value is a set of masking coordinates.
    
    # Store all individual areas in dict

    all_individual_ocl_areas_dict = {} # key = image_name, value = list of individual ocl areas. 

    for keys,values in (split_string.items()):
        pixel_area_list = [] # The list containing the pixel area calculated by the shoelace formula
        file_key = []
        for i in range(len(values)):
            if i != 0 and i != (len(values)-1):
                coordinate_list = values[i].split(',')     
                coordinate_list_as_floats = [] # New list is now a list of coordinates as a float
                for i in coordinate_list[5:]: # This will exclude the box coordinates and object score
                    if len(coordinate_list[5:]) >= 6: #This will make sure that the area is at least a three sided polygon. 
                        flt = float(i)
                        coordinate_list_as_floats.append(flt)
                    else:
                        continue

                # This conditional statement will make sure that every coordinate list has floats
                if len(coordinate_list_as_floats) >= 6 :
                    pixel_area = calculate_pixel_area(coordinate_list_as_floats)
                pixel_area_list.append(pixel_area)

        # create dict of image_name as keys and pixel area of all the individual ocl areas as values. 
        all_individual_ocl_areas_dict[keys] = pixel_area_list

        # Below will determine the total area of ocls in each well in pixels.
        # This will calculate total area of ocls in pixels
        total_area = (total_area_per_well(pixel_area_list))

        # Below was utilized to provide user with total area in pixels.
        if well_area_in_pixels != 0:
            percent_ocl_each_well = percent_ocl_area_per_well(total_area, well_area_in_pixels)
        
            write_area_to_output(img_dir, total_area, percent_ocl_each_well, out_dir, keys)
        
        else:
            percent_ocl_each_well = "None"
            write_area_to_output(img_dir, total_area, percent_ocl_each_well, out_dir, keys)

            print("If you want total area calculated, please enter pixel area of each well into the --total_well_area_in_pixels argument.")

    
    # function to create csv with all individual areas. 
    write_individual_ocl_area_to_output(img_dir, all_individual_ocl_areas_dict)

    return

if __name__ == '__main__':
    main(sys.argv)
