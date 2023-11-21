import json
import os
import numpy as np
from tqdm import tqdm
import time

from datasets.build_knowledge.helper import *


def gatther_all_frame_S3D_embeddings(args, logger):
    start_time = time.time()
    
    # get all video IDs
    # TODO: to account for cae vids list
    # videos = get_all_video_ids(args, logger)
    videos = get_all_video_ids(args, logger, format='txt')

    
    frame_embeddings = []
    frame_lookup_table = []
    
    videos_missing_features = set()
    for v in tqdm(videos):
        try:
            video_s3d = np.load(os.path.join(args.video_resource_dir, 'videos/s3d_visual_feats', '{}.npy'.format(v)))
            # video_s3d shape: (num_clips, num_subclips, 512)
            
            for c_idx in range(video_s3d.shape[0]):
                frame_embeddings.append(np.float64(np.mean(video_s3d[c_idx], axis=0)))
                frame_lookup_table.append((v, c_idx))

        except FileNotFoundError:
            videos_missing_features.add(v)

    logger.info("number of videos missing visual S3D features: {}".format(
        len(videos_missing_features)))
    if len(videos_missing_features) > 0:
        with open('videos_missing_features.pickle', 'wb') as f:
            pickle.dump(videos_missing_features, f)
    assert len(videos_missing_features) == 0, ("There are videos missing features! "
                                               + "Please check saved videos_missing_features.pickle.")
   
    frame_embeddings = np.array(frame_embeddings)
    
    logger.info("segment frame embeddings shape: {}".format(frame_embeddings.shape))
    # segment video embeddings shape: (3741608, 512) for the subset
    # segment video embeddings shape: (51053844, 512) for the fullset
    logger.info("getting all segment frame embeddings took {} s".format(round(time.time()-start_time, 2)))
    return frame_embeddings, frame_lookup_table


def gatther_all_caption_S3D_embeddings(args, logger, selective_video_ids=None):
    start_time = time.time()
    
    # get all video IDs and their corresponding video segments sorted by the temporal order
    if selective_video_ids is None:
        videos = get_all_video_ids(args, logger, format='txt')
    else:
        videos = selective_video_ids
    logger.info("number of videos: {}".format(len(videos)))

    caption_embeddings = []
    caption_lookup_table = []
    
    caption_missing_features = set()
    
    for v in tqdm(videos):
        try:
            caption_s3d = np.load(os.path.join(args.video_resource_dir,
                                               'subtitles/s3d_text_feats', '{}.npy'.format(v)))
            # caption_s3d shape: (num_captions, 512)
            for c_idx in range(caption_s3d.shape[0]):
                caption_embeddings.append(np.float64(caption_s3d[c_idx]))
                caption_lookup_table.append((v, c_idx))
    
        # except FileNotFoundError or ValueError:
        #     caption_missing_features.add(v)
        except:
            caption_missing_features.add(v)

    logger.info("number of videos missing textual S3D features: {}".format(len(caption_missing_features)))
    if len(caption_missing_features) > 0:
        with open('captions_missing_features.pickle', 'wb') as f:
            pickle.dump(caption_missing_features, f)
            logger.info("There are caption missing features!" + "Please check saved captions_missing_features.pickle.")
            
    caption_embeddings = np.array(caption_embeddings)
    logger.info("segment caption embeddings shape: {}".format(caption_embeddings.shape))
    logger.info("getting all segment caption embeddings took {} s".format(round(time.time() - start_time, 2)))
    return caption_embeddings, caption_lookup_table


def gather_wikihow_step_embeddings(args, logger, topic, wikihow_task_ids):
    step_embeddings = []
    step_lookup_table = []
    
    articles_missing_features = set()
    for task_id in wikihow_task_ids:
        try:
            steps_s3d = np.load(os.path.join(args.wikihow_dir,
                                             'step_headlines/s3d_text_feats/',
                                             '{}.npy'.format(task_id)))
            # steps_s3d shape: (num_steps, 512)
            for c_idx in range(steps_s3d.shape[0]):
                step_embeddings.append(np.float64(steps_s3d[c_idx]))
                step_lookup_table.append((task_id, c_idx))

        except FileNotFoundError:
            articles_missing_features.add(task_id)
    logger.info("number of articles that have missing step headlines' S3D textual features: {}".format(
        len(articles_missing_features)))

    if len(articles_missing_features) > 0:
        with open('topic{}_articles_missing_features.pickle'.format(topic), 'wb') as f:
            pickle.dump(articles_missing_features, f)
        logger.info("There are step headlines missing features!" + "Please check saved articles_missing_features.pickle.")

    step_embeddings = np.array(step_embeddings)
    return step_embeddings, step_lookup_table
    
    
def find_step_similarities_for_segments_using_frame(
    args, logger,
    step_des_feats, segment_video_embeddings, segment_video_lookup_table):
    
    start = time.time()

    for segment_id in tqdm(range(len(segment_video_embeddings))):
        v, cidx = segment_video_lookup_table[segment_id]
        save_path = os.path.join(args.video_resource_dir, 'videos/sim_scores/global', v, 'segment_{}.npy'.format(cidx))
        if not os.path.exists(save_path):
            # dot product as similarity score
            sim_scores = np.einsum('ij,ij->i',
                                   step_des_feats, 
                                   segment_video_embeddings[segment_id][np.newaxis, ...])
            
            os.makedirs(os.path.join(args.video_resource_dir, 'videos/sim_scores/global', v), exist_ok=True)
            np.save(save_path, sim_scores)
            
    logger.info('finding step similarity scores for segments using frames ' + 
                'took {} seconds'.format(time.time() - start))
    # os._exit(0)
    return


def find_step_similarities_for_segments_using_caption(
        args, logger,
        step_des_feats, segment_caption_embeddings, segment_video_lookup_table):
    
    start = time.time()

    for segment_id in tqdm(range(len(segment_caption_embeddings))):
        v, cidx = segment_video_lookup_table[segment_id]
        save_path = os.path.join(args.video_resource_dir, 'subtitles/sim_scores/{}'.format(args.graph_structure), v,
                                 'segment_caption_{}.npy'.format(cidx))
        if not os.path.exists(save_path):
            # dot product as similarity score
            sim_scores = np.einsum('ij,ij->i',
                                   step_des_feats,
                                   segment_caption_embeddings[segment_id][np.newaxis, ...])
            
            os.makedirs(os.path.join(args.video_resource_dir, 'subtitles/sim_scores/{}'.format(args.graph_structure), v),
                        exist_ok=True)
            np.save(save_path, sim_scores)
    
    logger.info(
        'finding step similarity scores for segments using captions ' +
        'took {} seconds'.format(time.time() - start))
    # os._exit(0)
    return


def get_sim_scores(args, logger):
    if args.use_captions:
        if args.graph_structure == 'global':
            segment_caption_embeddings, segment_caption_lookup_table = \
                gatther_all_caption_S3D_embeddings(args, logger)
            
            step_des_feats = get_step_des_feats(args, logger, language_model="S3D")
            # shape: (10, 588, 512)
            
            # TODO: re-factor the function, for WikiHow_FULL, there is no step_des_feats
            # (1) concatenate article_idx.npy
            find_step_similarities_for_segments_using_caption(
                    args, logger,
                    step_des_feats, segment_caption_embeddings, segment_caption_lookup_table)
            
        elif args.graph_structure == 'local':
            with open(os.path.join(args.wikihow_dir, 'task_titles/{}/topic2task.pickle'.format(args.video_resource)), 'rb') as f:
                topic_2_taskids = pickle.load(f)

            task_id_to_video_ids, video_id_to_task_id = get_task_ids_to_video_ids(args)

            topic2step_lookup_table = {}
            for topic, clustered_taskids in topic_2_taskids.items():
                # find the wikihow task ids
                wikihow_taskids = [task_id for (source, task_id) in clustered_taskids if source == 'wikihow']
                if len(wikihow_taskids) > 0:
                    logger.info('Processing topic {}...'.format(topic))
                    step_des_feats, step_lookup_table = \
                        gather_wikihow_step_embeddings(args, logger, topic, wikihow_taskids)
                    topic2step_lookup_table[topic] = step_lookup_table
                    
                    # find the video resource task ids
                    video_taskids = [task_id for (source, task_id) in clustered_taskids if source == 'howto100m']
                    # from video taskids to video ids:
                    relevant_video_ids = [video_id for task_id in video_taskids for video_id in task_id_to_video_ids.get(task_id)]
                    segment_caption_embeddings, segment_caption_lookup_table = \
                        gatther_all_caption_S3D_embeddings(args, logger, relevant_video_ids)
                    
                    find_step_similarities_for_segments_using_caption(
                            args, logger, step_des_feats, segment_caption_embeddings, segment_caption_lookup_table)
                else:
                    logger.info('There is no wikihow tasks clustered in topic {}'.format(topic))

            with open(os.path.join(args.wikihow_dir, 'task_titles/{}/topic2step.pickle'.format(args.video_resource)), 'wb') as f:
                pickle.dump(topic2step_lookup_table, f)
    else:
        # TODO: write global and local --> condense these two blocks into one
        segment_video_embeddings, segment_video_lookup_table = \
            gatther_all_frame_S3D_embeddings(args, logger)
    
        find_step_similarities_for_segments_using_frame(
                args, logger,
                step_des_feats, segment_video_embeddings, segment_video_lookup_table)
