"""
This script refines pose estimates stored in a CSV file using the MegaPose refiner model.
The input CSV file should be in BOP Toolkit results format.
The refined poses are saved to a new CSV file in the same directory as the input file,
with "-megapose-refined" appended to the method name in the filename.

Example usage:
python happypose/pose_estimators/megapose/scripts/run_refinement_from_pose_csv.py ipt_foundpose/coarse-cam0_itoddmv-test.csv  --sensor cam0 --n_best 50
"""


import argparse
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from bop_toolkit_lib.dataset_params import get_split_params, scene_tpaths_keys
from bop_toolkit_lib.config import datasets_path, results_path
from bop_toolkit_lib.inout import save_bop_results, load_bop_results, parse_result_filename, load_scene_camera, create_pose_result_filename
import imageio.v2 as iio

from happypose.toolbox.inference.types import ObservationTensor
from happypose.toolbox.utils.load_model import load_named_model
from happypose.toolbox.inference.types import PoseEstimatesType
from happypose.toolbox.datasets.datasets_cfg import make_object_dataset




def reorganize_results(data):
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for entry in data:
        scene = entry['scene_id']
        im = entry['im_id']
        
        # Append values to nested lists
        result[scene][im]['obj_id'].append(entry['obj_id'])
        result[scene][im]['score'].append(entry['score'])
        result[scene][im]['R'].append(entry['R'])
        result[scene][im]['t'].append(entry['t'].reshape(-1))  # flatten (3,1) → (3,)
        result[scene][im]['time'].append(entry['time'])
    
    # Convert lists of arrays → stacked numpy arrays
    for scene_dict in result.values():
        for im_dict in scene_dict.values():
            im_dict['R'] = np.stack(im_dict['R'])
            im_dict['t'] = np.stack(im_dict['t'])
    
    return result


def load_im2rgb(
        path: str | Path, 
        ds_name:  None | str = None, 
        sensor: None | str = None) -> np.ndarray:
    """Loads an image from a file as numpy array.
    
    If original image is grayscale, an RGB image is created
    by tiling the grayscale image.

    :param path: Path to the image file to load.
    :return: ndarray with the loaded image, (w,h,3).
    """
    img = iio.imread(path)
    if ds_name is not None and ds_name == "itoddmv":
        if sensor is not None and sensor.startswith("cam"):
            # additional itoddmv cam* images are uint8, storing information in the 12 most significant bits
            img = (img >> 4).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img


parser = argparse.ArgumentParser()
parser.add_argument("result_name")
parser.add_argument("--results_path", default=results_path)
parser.add_argument("--sensor", default=None)
parser.add_argument("--modality", default=None)
parser.add_argument("--device", default="cuda:0")
parser.add_argument("--n_refiner_iterations", type=int, default=5)
parser.add_argument("--n_best", type=int, default=-1, help="If >0, only refine this many best scoring poses per image.")
args = parser.parse_args()

modality = args.modality
sensor = args.sensor
results_path = args.results_path
DATASETS_FIXED_NORMALS = ["itodd", "itoddmv"]

print("results_path:", results_path)


# loading the csv
csv_path = Path(results_path) / args.result_name
print(f"Reading {csv_path}")
result_name, method, dataset, split, split_type, ext = parse_result_filename(csv_path)
results = load_bop_results(csv_path)
results = reorganize_results(results)

# load megapose models
object_dataset = make_object_dataset(dataset)
pose_estimator = load_named_model(
    model_name="megapose-1.0-RGB",
    object_dataset=object_dataset,
    n_workers=8,
    bsz_images=128,
)

# loading dataset params
dp_split = get_split_params(datasets_path, dataset, split)


out_results = []

for scene_id in tqdm(results, colour="green", desc="Scenes"):
    if dp_split["eval_sensor"] is None and dp_split["eval_modality"] is None:
        tpath_keys = scene_tpaths_keys(None, None, scene_id)
    else:
        modality = dp_split["eval_modality"] if modality is None else modality
        sensor = dp_split["eval_sensor"] if sensor is None else sensor
        assert modality is not None and sensor is not None
        tpath_keys = scene_tpaths_keys(modality, sensor, scene_id)
    scene_camera = load_scene_camera(dp_split[tpath_keys["scene_camera_tpath"]].format(scene_id=scene_id))

    for im_id in tqdm(results[scene_id], colour="blue", desc="Images", leave=False):

        if dataset == "itodd":
            im_path = dp_split["gray_tpath"].format(scene_id=scene_id, im_id=im_id)
        else:
            im_path = dp_split[tpath_keys["rgb_tpath"]].format(scene_id=scene_id, im_id=im_id)

        rgb = load_im2rgb(im_path, ds_name=dataset, sensor=sensor)
        K = scene_camera[im_id]["cam_K"]
        observation = ObservationTensor.from_numpy(rgb, None, K).to(args.device)

        # create a PoseEstimatesType from the loaded results 
        im_est = results[scene_id][im_id]
        df_infos = pd.DataFrame({
            'batch_im_id': [0]*len(im_est['obj_id']),
            'label': [f"{dataset}-obj_{obj_id:06d}" for obj_id in im_est['obj_id']],
            'instance_id': list(range(len(im_est['obj_id']))),
            'score': im_est['score'],
        })
        poses = torch.zeros((len(im_est['obj_id']), 4, 4), dtype=torch.float32)
        poses[:, :3, :3] = torch.from_numpy(im_est['R'].astype(np.float32))
        poses[:, :3, 3] = torch.from_numpy(im_est['t'].astype(np.float32))
        poses[:, :3, 3] *= 0.001  # mm → m
        data_TCO_input = PoseEstimatesType(df_infos, poses=poses)
        data_TCO_input.to(args.device)
        if args.n_best > 0:
            ids_best = (-data_TCO_input.infos.score).argsort()[:args.n_best]
            data_TCO_input = data_TCO_input[ids_best]

        # filter out unrealistic poses based on object size and distance to camera
        outlier_indices = data_TCO_input.poses[:,:3,3].norm(dim=1) > 100  # 100m
        indices_ok = data_TCO_input.infos[(~outlier_indices.cpu()).tolist()].index.to_list()
        data_TCO_input = data_TCO_input[indices_ok]

        # Refine the poses using the refiner model.        
        timing_str = ""
        preds, refiner_extra_data = pose_estimator.forward_refiner(
            observation,
            data_TCO_input,
            n_iterations=args.n_refiner_iterations,
        )
        data_TCO_refined = preds[f"iteration={args.n_refiner_iterations}"]
        timing_str += f"refiner={refiner_extra_data['time']:.2f}, "

        # Score the refined poses using the coarse model.
        data_TCO_scored, scoring_extra_data = pose_estimator.forward_scoring_model(
            observation,
            data_TCO_refined,
        )
        timing_str += f"scoring={scoring_extra_data['time']:.2f}, "

        poses_arr = data_TCO_scored.poses.cpu().numpy()
        R_arr = poses_arr[:, :3, :3]
        t_arr = poses_arr[:, :3, 3:]
        # total time: initial estimate (same for alll detections) + refiner + scoring
        t_im_est_refined = im_est["time"][0] + refiner_extra_data['time'] + scoring_extra_data['time']
        print("{len(data_TCO_scored):", len(data_TCO_scored), "t_im_est_refined", t_im_est_refined, "t_im_est:", im_est["time"][0], "t_refiner:", refiner_extra_data['time'], "t_scoring:", scoring_extra_data['time'])
        for i in range(len(data_TCO_scored)):
            obj_id = int(data_TCO_scored.infos.iloc[i]['label'].split('-obj_')[-1])
            score = float(data_TCO_scored.infos.iloc[i]['pose_score'])
            result = {
                "scene_id": scene_id,
                "im_id": im_id,
                "obj_id": obj_id,
                "score": score,
                "R": R_arr[i],
                "t": t_arr[i]*1000.0,  # m → mm
                "time": t_im_est_refined,
            }
            out_results.append(result)

out_file_name = create_pose_result_filename(
    method=method + "-megapose-refined",
    dataset=dataset,
    split=split,
    split_type=split_type,
    optional_id=sensor if sensor is not None else None,
)
print(f"Saving refined results to {csv_path.parent / out_file_name}")
save_bop_results(csv_path.parent / out_file_name, out_results)

# fast exit to avoid hanging processes
import os
os._exit(0)