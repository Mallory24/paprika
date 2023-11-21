import math
import os
import numpy as np
import glob
import json
from tqdm import tqdm
from collections import defaultdict
from itertools import repeat
from multiprocessing import Pool
from scipy.sparse import csr_matrix

from datasets.build_knowledge.helper import *

import sys


def get_num_neighbors_of_nodes(G):
	"""
	- G: an nxn array to represent adj matrx
	"""
	num_neighbors = []
	for i in range(len(G)):
		num_neighbors.append(len(np.where(G[i] > 0)[0]))
	return num_neighbors


def threshold_and_normalize(args, logger, G, edge_min_aggconf=1000):
	logger.info('thresholding edges...')
	G_new = np.zeros((G.shape[0], G.shape[0]))
	for i in range(G.shape[0]):
		for j in range(G.shape[0]):
			if G[i, j] > edge_min_aggconf:
				G_new[i, j] = G[i, j]
	G = G_new
	
	G_flat = G.reshape(G.shape[0] * G.shape[0], )
	x = [np.log(val) for val in G_flat if val != 0]
	# TODO: change it in order to
	assert len(x) > 0, 'No edges remain after thresholding! Please use a smaller edge_min_aggconf!'
	max_val, min_val = np.max(x), 0
	
	logger.info('normalizing edges...')
	G_new = np.zeros((G.shape[0], G.shape[0]))
	for i in range(G.shape[0]):
		for j in range(G.shape[0]):
			if G[i, j] > 0:
				G_new[i, j] = (np.log(G[i, j]) - 0) / (max_val - 0)  # log min max norm
	G = G_new
	return G


def get_edges_between_wikihow_steps_in_wikihow(args, logger, wikihow_taskids):
	with open(os.path.join(args.wikihow_dir, 'step_label.json'), 'r') as f:
		wikihow = json.load(f)

	step_id = 0
	article_id_to_step_id = defaultdict()
	for article_id in wikihow_taskids:
		for article_step_idx in range(len(wikihow[article_id])):
			article_id_to_step_id[(article_id, article_step_idx)] = step_id
			step_id += 1
	total_num_steps = len(article_id_to_step_id)
	
	wikihow_steps_1hop_edges = np.zeros((total_num_steps, total_num_steps))
	for article_id in wikihow_taskids:
		for article_step_idx in range(1, len(wikihow[article_id])):
			predecessor = article_id_to_step_id[(article_id, article_step_idx - 1)]
			successor = article_id_to_step_id[(article_id, article_step_idx)]
			
			wikihow_steps_1hop_edges[predecessor, successor] += 1
	
	return wikihow_steps_1hop_edges


def get_edges_between_wikihow_steps_of_one_howto100m_video(args, video, total_num_steps, sim_score_path):
	sim_score_paths_of_segments_this_video = sorted(glob.glob(os.path.join(sim_score_path, video, 'segment_*.npy')))
	
	edges_meta = list()
	# loop over segments
	for video_segment_idx in range(1, len(sim_score_paths_of_segments_this_video)):
		segment_pre_sim_scores = np.load(sim_score_paths_of_segments_this_video[video_segment_idx - 1])
		segment_suc_sim_scores = np.load(sim_score_paths_of_segments_this_video[video_segment_idx])
		
		# the similarity score is only computed on a local level
		assert len(segment_pre_sim_scores) == len(segment_suc_sim_scores) == total_num_steps
		
		predecessors, _ = find_matching_of_a_segment(segment_pre_sim_scores,
			criteria=args.graph_find_matched_steps_criteria,
			threshold=args.graph_find_matched_steps_for_segments_thresh,
			topK=args.graph_find_matched_steps_for_segments_topK)

		successors, _ = find_matching_of_a_segment(segment_suc_sim_scores,
			criteria=args.graph_find_matched_steps_criteria,
			threshold=args.graph_find_matched_steps_for_segments_thresh,
			topK=args.graph_find_matched_steps_for_segments_topK)

		for predecessor in predecessors:
			for successor in successors:
				if predecessor != successor:  # a step transition
					edges_meta.append([predecessor, successor,
					                   segment_pre_sim_scores[predecessor] * segment_suc_sim_scores[successor]])
	
	print(f'done processing video {video}', flush=True)
	return edges_meta


def get_edges_between_wikihow_steps_in_howto100m(args, logger, total_num_steps, howto100m_taskids):
	if args.use_captions:
		sim_score_path = os.path.join(args.video_resource_dir, 'subtitles/sim_scores/local/')
	else:
		sim_score_path = os.path.join(args.video_resource_dir, 'videos/sim_scores/local/')
	
	# on a selective set of videos whose task ids are clustered into the same topic
	taskid_to_videoid, _ = get_task_ids_to_video_ids(args)
	matched_videos = [video for taskid in howto100m_taskids for video in taskid_to_videoid.get(taskid)]
	
	chunksize = math.ceil(len(matched_videos) / args.num_workers)
	logger.info('use multiprocessing with chunksize {} to get edges between wikihow step headlines from howto100m...'.format(chunksize))
	# logger.info('use multiprocessing to get edges between wikihow step headlines from howto100m...')
	
	with Pool(processes=args.num_workers) as pool:
		edges_metas = pool.starmap(get_edges_between_wikihow_steps_of_one_howto100m_video,
		                           zip(repeat(args), matched_videos, repeat(total_num_steps), repeat(sim_score_path)),
		                           chunksize=chunksize)
	
	howto100m_steps_1hop_edges = np.zeros((total_num_steps, total_num_steps))
	for edges_meta in edges_metas:
		for [predecessor, successor, confidence] in edges_meta:
			howto100m_steps_1hop_edges[predecessor, successor] += confidence

	logger.info('multiprocessing finished! going to threshold and normalize edges...')
	howto100m_steps_1hop_edges = threshold_and_normalize(args, logger, howto100m_steps_1hop_edges,
	                                                     args.edge_min_aggconf)
	
	return howto100m_steps_1hop_edges


def get_node_transition_candidates(args, logger, step2node, G_wikihow, G_howto100m):
	# TODO: To use this, one has to adapt step2node on the local graph: topic => corresponding steps => clustering => nodes
	candidates = defaultdict(list)
	
	if args.graph_use_wikihow:
		for step_id in tqdm(range(len(step2node))):
			for direct_outstep_id in G_wikihow[step_id].indices:
				conf = G_wikihow[step_id, direct_outstep_id]
				
				node_id = step2node[step_id]
				direct_outnode_id = step2node[direct_outstep_id]
				
				candidates[(node_id, direct_outnode_id)].append(conf)
		
		logger.info('collected node transition candidates (len: {}) from wikiHow...'.format(len(candidates)))
	else:
		# TODO: write it also on get_edges.py
		logger.info('disable collecting node transition candidates from wikiHow...')
	
	for step_id in tqdm(range(len(step2node))):
		for direct_outstep_id in G_howto100m[step_id].indices:
			conf = G_howto100m[step_id, direct_outstep_id]
			
			node_id = step2node[step_id]
			direct_outnode_id = step2node[direct_outstep_id]
			
			candidates[(node_id, direct_outnode_id)].append(conf)
	
	logger.info('collected node transition candidates (len: {}) from howTo100M...'.format(len(candidates)))
	
	return candidates


def get_step_transition_candidates(args, logger, G_wikihow, G_howto100m):
	candidates = defaultdict(list)
	num_steps = G_wikihow.shape[0]

	if args.graph_use_wikihow:
		for step_id in tqdm(range(num_steps)):
			for direct_outstep_id in G_wikihow[step_id].indices:
				conf = G_wikihow[step_id, direct_outstep_id]
				candidates[(step_id, direct_outstep_id)].append(conf)
		logger.info('collected node transition candidates (len: {}) from wikiHow...'.format(len(candidates)))
	else:
		# TODO: write it also on get_edges.py
		logger.info('disable collecting node transition candidates from wikiHow...')
	
	for step_id in tqdm(range(num_steps)):
		for direct_outstep_id in G_howto100m[step_id].indices:
			conf = G_howto100m[step_id, direct_outstep_id]
			candidates[(step_id, direct_outstep_id)].append(conf)
	logger.info('collected node transition candidates (len: {}) from howTo100M...'.format(len(candidates)))
	
	return candidates


def keep_highest_conf_for_each_candidate(args, logger, candidates):
	edges = defaultdict()
	for (node_id, direct_outnode_id) in tqdm(candidates):
		max_conf = np.max(candidates[(node_id, direct_outnode_id)])
		
		edges[(node_id, direct_outnode_id)] = max_conf
	logger.info(
		'kept only the highest conf score for each node transition candidate... len(edges): {}'.format(len(edges)))
	return edges


def build_pkg_adj_matrix(edges, num_nodes):
	pkg = np.zeros((num_nodes, num_nodes))
	for (node_id, direct_outnode_id) in tqdm(edges):
		pkg[node_id, direct_outnode_id] = edges[(node_id, direct_outnode_id)]
	return pkg


def get_edges(args, logger, topic, wikihow_taskids, howto100m_taskids):
	# save the output
	if args.use_captions:
		graph_savedir = os.path.join(args.video_resource_dir, 'subtitles/graph_output/local/topic_{}'.format(topic))
	else:
		graph_savedir = os.path.join(args.video_resource_dir, 'videos/graph_output/local/topic_{}'.format(topic))
	os.makedirs(graph_savedir, exist_ok=True)
	
	pkg_savepath = os.path.join(graph_savedir,
		'PKG-criteria_{}-threshold_{}-topK_{}-agg_{}.npy'.format(args.graph_find_matched_steps_criteria,
			args.graph_find_matched_steps_for_segments_thresh, args.graph_find_matched_steps_for_segments_topK,
			args.edge_min_aggconf))
	
	G_wikihow_savepath = os.path.join(graph_savedir,
		'G_wikihow-criteria_{}-threshold_{}-topK_{}.npy'.format(args.graph_find_matched_steps_criteria,
			args.graph_find_matched_steps_for_segments_thresh, args.graph_find_matched_steps_for_segments_topK))
	
	G_howto100m_savepath = os.path.join(graph_savedir,
		'G_howto100m-criteria_{}-threshold_{}-topK_{}-agg_{}.npy'.format(args.graph_find_matched_steps_criteria,
			args.graph_find_matched_steps_for_segments_thresh, args.graph_find_matched_steps_for_segments_topK,
			args.edge_min_aggconf))
	
	if not os.path.exists(pkg_savepath) and not os.path.exists(G_howto100m_savepath):
		logger.info('get local graph constructed based on topic (cluster) {}'.format(topic))
		# --  get the edges between step headlines
		logger.info('get edges between wikihow step headlines in wikihow...')
		G_wikihow = get_edges_between_wikihow_steps_in_wikihow(args, logger, wikihow_taskids)
		# num_neighbors = get_num_neighbors_of_nodes(G_wikihow)
		
		logger.info('get edges between wikihow step headlines in howto100m...')
		G_howto100m = get_edges_between_wikihow_steps_in_howto100m(args, logger, G_wikihow.shape[0], howto100m_taskids)
		# num_neighbors = get_num_neighbors_of_nodes(G_howto100m)
		
		G_wikihow_csr, G_howto100m_csr = csr_matrix(G_wikihow), csr_matrix(G_howto100m)
		
		# 12.Oct. 2023: to quickly fix the scope of local graph, disable node transitions
		# -- turn edges between step headlines into edges between nodes
		# logger.info('turn edges between step headlines into edges between nodes...')
		# from datasets.build_knowledge.get_nodes import get_nodes
		# node2step, step2node = get_nodes(args, logger)
		#
		# node_transition_candidates = get_node_transition_candidates(args, logger, num_of_steps=G_wikihow.shape[0], G_wikihow=None, G_howto100m=G_howto100m_csr)
		# pkg_edges = keep_highest_conf_for_each_candidate(args, logger, node_transition_candidates)
		# pkg = build_pkg_adj_matrix(pkg_edges, len(node2step))
		
		# 12.Oct. 2023: to quickly fix the scope of local graph, disable node transitions
		step_transition_candidates = get_step_transition_candidates(args, logger, G_wikihow_csr, G_howto100m_csr)
		pkg_edges = keep_highest_conf_for_each_candidate(args, logger, step_transition_candidates)
		
		pkg = build_pkg_adj_matrix(pkg_edges, num_nodes=G_wikihow.shape[0])
		
		logger.info('pkg built!')
		
		with open(pkg_savepath, 'wb') as f:
			np.save(f, pkg)
		logger.info('{} saved!'.format(pkg_savepath))
	
		if args.graph_use_wikihow:
			with open(G_wikihow_savepath, 'wb') as f:
				np.save(f, G_wikihow)
			logger.info('{} saved!'.format(G_wikihow_savepath))
	
		with open(G_howto100m_savepath, 'wb') as f:
			np.save(f, G_howto100m)
		logger.info('{} saved!'.format(G_howto100m_savepath))
	else:
		logger.info('local graph based on topic (cluster) {} has been constructed'.format(topic))

	
def get_local_graph_based_on_topic_similarity(args, logger):
	# iterate through the topic
	with open(os.path.join(args.wikihow_dir, 'task_titles/{}/topic2task.pickle'.format(args.video_resource)), 'rb') as f:
		topic2task = pickle.load(f)

	meaningful_local_graphs = []
	for topic, grouped_tasks in topic2task.items():
		wikihow_taskids = [task_id for source, task_id in grouped_tasks if source == 'wikihow']
		howto100m_taskids = [task_id for source, task_id in grouped_tasks if source == 'howto100m']
		
		if len(wikihow_taskids) > 0 and len(howto100m_taskids) > 0:
			meaningful_local_graphs.append((topic, wikihow_taskids, howto100m_taskids))
	
	logger.info('there are {} local graphs to be constructed'.format(len(meaningful_local_graphs)))
	
	for (topic, wikihow_taskids, howto100m_taskids) in meaningful_local_graphs:
		# Nov. 20: for quick test, select the cluster
		if topic in [11, 101, 133, 147, 258]:
			get_edges(args, logger, topic, wikihow_taskids, howto100m_taskids)
