import os
import time
import numpy as np
import json
from tqdm import tqdm
import pickle
import pandas as pd
from collections import defaultdict


def get_step_des_feats(args, logger, language_model='MPNet'):
    """
    language_model: 'MPNet' or 'S3D'
    """
    
    if language_model == 'MPNet':
        step_des_feats = np.load(
            os.path.join(args.wikihow_dir, 'mpnet_feat.npy'))
    
    elif language_model == 'S3D':
        
        with open(
            os.path.join(
                args.wikihow_dir, 
                'step_headlines/s3d_text_feat/step_embeddings.pickle'
            ), 'rb') as f:
            
            step_des_feats = pickle.load(f)
        
    return step_des_feats


def get_task_titles_feats(input_dir):
    with open(os.path.join(input_dir,
                           'task_titles/s3d_text_feats/task_title_embeddings.pickle'), 'rb') as f:
        task_title_feats = pickle.load(f)

    return task_title_feats


def merge_task_title_feats(wikihow_tasktitle_feat_dict, howto100m_tasktitle_feat_dict):
    merged_task_id_2_original_task_id = defaultdict()
    merged_feats = []
    
    for idx, (article_id, tasktitle_feat) in enumerate(wikihow_tasktitle_feat_dict.items()):
        merged_task_id_2_original_task_id[idx] = ('wikihow', article_id)
        merged_feats.append(tasktitle_feat)
        
    for idx, (article_id, tasktitle_feat) in enumerate(howto100m_tasktitle_feat_dict.items()):
        idx += len(wikihow_tasktitle_feat_dict)
        merged_task_id_2_original_task_id[idx] = ('howto100m', article_id)
        merged_feats.append(tasktitle_feat)
        
    merged_tasktitle_feats = np.concatenate(merged_feats, axis=0)

    return merged_task_id_2_original_task_id, merged_tasktitle_feats


def get_all_video_ids(args, logger, format=None):
    start_time = time.time()
    if format == 'txt':
        video_path = os.path.join(args.video_resource_dir, 'vids.txt')
        with open(video_path, 'r') as f:
            videos = [vid.replace('\n', '') for vid in f.readlines()]
    
    # TODO: put used videos ids in a *.txt, and delete the following code block
    else:
        # Changed: args.howto100m_dir to args.frame_dir
        if os.path.exists(os.path.join(args.frame_dir, 'video_IDs.npy')):
            videos = np.load(os.path.join(args.frame_dir, 'video_IDs.npy'))
        else:
            videos = []
            for f in tqdm(os.listdir(os.path.join(args.frame_dir, 'feats'))):
                if os.path.isdir(os.path.join(args.frame_dir, 'feats', f)):
                    if os.path.exists(os.path.join(args.frame_dir, 'feats', f, 'text_mpnet.npy')):
                        # raw_captions.pickle  segment_time.npy  status.txt  text_mpnet.npy  text.npy  video.npy
                        videos.append(f)
            logger.info("number of videos: {}".format(len(videos)))
            np.save(os.path.join(args.howto100m_dir, 'video_IDs.npy'), videos)
        
    logger.info("getting all video IDs took {} s".format(round(time.time()-start_time, 2)))
    return videos


def get_task_ids_to_video_ids(args):
    task_id_to_video_ids = {}
    video_id_to_task_id = {}

    video_meta_df = pd.read_csv(args.video_meta_csv_path)
  
    for index, row in video_meta_df.iterrows():
        video_id = row['video_id']
        task_id = row['task_id']
        if task_id not in task_id_to_video_ids:
            task_id_to_video_ids[task_id] = []
        task_id_to_video_ids[task_id].append(video_id)
    
    for task_id, video_ids in task_id_to_video_ids.items():
        for video_id in video_ids:
            video_id_to_task_id[video_id] = task_id
    
    return task_id_to_video_ids, video_id_to_task_id


def find_matching_of_a_segment(
    sim_scores, criteria="threshold", threshold=0.7, topK=3):
    
    sorted_values = np.sort(sim_scores)[::-1]  # sort in descending order
    sorted_indices = np.argsort(-sim_scores)  # indices of sorting in descending order

    matched_steps, matched_steps_score = find_matching_of_a_segment_given_sorted_val_corres_idx(
        sorted_values, sorted_indices, criteria=criteria, threshold=threshold, topK=topK)
    
    return matched_steps, matched_steps_score


def find_matching_of_a_segment_given_sorted_val_corres_idx(
    sorted_values, sorted_indices, criteria="threshold", threshold=0.7, topK=3):
    
    matched_steps = list()
    matched_steps_score = list()

    if criteria == "threshold":
        # Pick all steps with sim-score > threshold.
        for i in range(len(sorted_values)):
            if sorted_values[i] > threshold:
                matched_steps.append(sorted_indices[i])
                matched_steps_score.append(sorted_values[i])
        
    elif criteria == "threshold+topK":
        # From the ones with sim-score > threshold, 
        # pick the top K if existing.
        for i in range(len(sorted_values)):
            if sorted_values[i] > threshold:
                if len(matched_steps) < topK:
                    matched_steps.append(sorted_indices[i])
                    matched_steps_score.append(sorted_values[i])
                else:
                    break
    
    elif criteria == "topK":
        # Pick the top K
        for i in range(len(sorted_indices)):
            if len(matched_steps) < topK:
                matched_steps.append(sorted_indices[i])
                matched_steps_score.append(sorted_values[i])
            else:
                break
                
    else:
        print('The criteria is not implemented!\nFunc: {}\nFile:{}'.format(
            __name__, __file__))
        os._exit(0)
    
    return matched_steps, matched_steps_score


