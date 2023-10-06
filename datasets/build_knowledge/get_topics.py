import os
import numpy as np
import time
import pickle
from collections import defaultdict

from datasets.build_knowledge.helper import *


def get_nodes_by_removing_step_duplicates(args, logger, wikihow_tasktitle_feats=None, video_resource_tasktitle_feats=None):
	start_time = time.time()
	if os.path.exists(os.path.join(args.wikihow_dir, 'task2topic.pickle')) and os.path.exists(
		os.path.join(args.wikihow_dir, 'topic2task.pickle')) and os.path.exists(
		os.path.join(args.task_dir, 'task2topic.pickle')):
		with open(os.path.join(args.wikihow_dir, 'task2topic.pickle'), 'rb') as f:
			wikihow_task2topic = pickle.load(f)
		with open(os.path.join(args.task_dir, 'task2topic.pickle'), 'rb') as f:
			howto100m_task2topic = pickle.load(f)
		with open(os.path.join(args.wikihow_dir, 'topic2task.pickle'), 'rb') as f:
			topic2task = pickle.load(f)
	else:
		assert wikihow_tasktitle_feats is not None
		assert howto100m_tasktitle_feats is not None
		from sklearn.cluster import AgglomerativeClustering
		
		with open(os.path.join(args.wikihow_dir, 'article_id_to_title.txt'), 'r') as f:
			wikihow_article_ids = [('wikihow', line.rstrip().split('\t')[0]) for line in f.readlines()]
		
		with open(os.path.join(args.task_dir, 'task_ids.csv'), 'r') as f:
			video_resource_task_ids = [('howto100m', line.rstrip().split('\t')[0]) for line in f.readlines()]
		
		merged_original_task_ids = wikihow_article_ids + video_resource_task_ids
		merged_task_feats = np.concatenate((wikihow_tasktitle_feats, video_resource_tasktitle_feats), axis=0)
		assert len(merged_original_task_ids) == len(merged_task_feats)
		merged_task_id_2_original_task_id = {merged_id: original_task_id for (merged_id, original_task_id)
		                                     in zip(range(len(merged_task_feats)), merged_original_task_ids)}
		
		clustering = AgglomerativeClustering(n_clusters=None, linkage=args.task_clustering_linkage,
			distance_threshold=args.task_clustering_distance_thresh, affinity=args.task_clustering_affinity).fit(
			merged_task_feats)
		# distance_threshold:
		#   The linkage distance threshold above which, clusters will not be merged.
		num_nodes = clustering.n_clusters_
		
		topic2task, wikihow_task2topic, howto100m_task2topic = defaultdict(), defaultdict(), defaultdict()

		for cluster_id in range(num_nodes):
			cluster_members = np.where(clustering.labels_ == cluster_id)[0]
			topic2task[cluster_id] = [merged_task_id_2_original_task_id[task_id] for task_id in cluster_members]
			for task_id in cluster_members:
				original_task_id = merged_task_id_2_original_task_id[task_id]
				if original_task_id[0] == 'wikihow':
					wikihow_task2topic[original_task_id[1]] = cluster_id
				else:
					howto100m_task2topic[original_task_id[1]] = cluster_id
		
		# for wikihow
		with open(os.path.join(args.wikihow_dir, 'topic2task.pickle'), 'wb') as f:
			pickle.dump(topic2task, f)
			
		with open(os.path.join(args.wikihow_dir, 'task2topic.pickle'), 'wb') as f:
			pickle.dump(wikihow_task2topic, f)
		
		# for video resources
		with open(os.path.join(args.task_dir, 'task2topic.pickle'), 'wb') as f:
			pickle.dump(howto100m_task2topic, f)

		logger.info("from task titles to topics took {} s".format(round(time.time() - start_time, 2)))

	return topic2task, wikihow_task2topic, howto100m_task2topic


def get_topics(args, logger):
	# load both wikiHow KB task titles and Howto100M or CAE task titles
	wikihow_tasktitle_feats, video_resource_tasktitle_feats = get_task_titles_feats(args.wikihow_dir, args.task_dir)
	
	topic2task, wikihow_task2topic, howto100m_task2topic = \
		get_topics_by_clustering_tasktitles(args, logger, wikihow_tasktitle_feats, video_resource_tasktitle_feats)
	return topic2task, wikihow_task2topic, howto100m_task2topic