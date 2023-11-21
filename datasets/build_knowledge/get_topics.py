import os
import numpy as np
import time
import pickle
from collections import defaultdict

from datasets.build_knowledge.helper import *


def get_topics_by_clustering_tasktitles(args, logger, wikihow_tasktitle_feats: dict, howto100m_tasktitle_feats: dict):
	start_time = time.time()
	if os.path.exists(os.path.join(args.wikihow_dir, 'task_titles/{}/task2topic.pickle'.format(args.video_resource))) \
			and os.path.exists(os.path.join(args.wikihow_dir, 'task_titles/{}/topic2task.pickle'.format(args.video_resource))) \
			and os.path.exists(os.path.join(args.video_resource_dir, 'task_titles/task2topic.pickle')):
		with open(os.path.join(args.wikihow_dir, 'task_titles/{}/task2topic.pickle'.format(args.video_resource)), 'rb') as f:
			wikihow_task2topic = pickle.load(f)
		with open(os.path.join(args.video_resource_dir, 'task_titles/task2topic.pickle'), 'rb') as f:
			howto100m_task2topic = pickle.load(f)
		with open(os.path.join(args.wikihow_dir, 'task_titles/{}/topic2task.pickle'.format(args.video_resource)), 'rb') as f:
			topic2task = pickle.load(f)
	else:
		assert wikihow_tasktitle_feats is not None
		assert howto100m_tasktitle_feats is not None
		from sklearn.cluster import AgglomerativeClustering
		
		merged_task_id_2_original_task_id, merged_task_feats = \
			merge_task_title_feats(wikihow_tasktitle_feats, howto100m_tasktitle_feats)
		
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
				original_tasktitle_source,  original_task_id = merged_task_id_2_original_task_id[task_id]
				if original_tasktitle_source == 'wikihow':
					wikihow_task2topic[original_task_id] = cluster_id
				else:
					howto100m_task2topic[original_task_id] = cluster_id
		
		# for wikihow
		with open(os.path.join(args.wikihow_dir, 'task_titles/{}/topic2task.pickle'.format(args.video_resource)), 'wb') as f:
			pickle.dump(topic2task, f)
			
		with open(os.path.join(args.wikihow_dir, 'task_titles/{}/task2topic.pickle'.format(args.video_resource)), 'wb') as f:
			pickle.dump(wikihow_task2topic, f)
		
		# for video resources
		with open(os.path.join(args.video_resource_dir, 'task_titles/topic2task.pickle'), 'wb') as f:
			pickle.dump(topic2task, f)
			
		with open(os.path.join(args.video_resource_dir, 'task_titles/task2topic.pickle'), 'wb') as f:
			pickle.dump(howto100m_task2topic, f)

		logger.info("Num of wikiHow tasks: {}".format(len(wikihow_task2topic)))
		logger.info("Num of video resource tasks: {}".format(len(howto100m_task2topic)))
		logger.info("Num of resulting topics: {}".format(len(topic2task)))
		logger.info("form task titles to topics took {} s".format(round(time.time() - start_time, 2)))

	return topic2task, wikihow_task2topic, howto100m_task2topic


def get_topics(args, logger):
	# load both wikiHow KB task titles and Howto100M or CAE task titles
	wikihow_tasktitle_feats = get_task_titles_feats(args.wikihow_dir)
	howto100m_tasktitle_feats = get_task_titles_feats(args.video_resource_dir)
	
	topic2task, wikihow_task2topic, howto100m_task2topic = \
		get_topics_by_clustering_tasktitles(args, logger, wikihow_tasktitle_feats, howto100m_tasktitle_feats)
	
	return topic2task, wikihow_task2topic, howto100m_task2topic