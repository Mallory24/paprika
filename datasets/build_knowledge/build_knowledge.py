import os

def obtain_external_knowledge(args, logger):
    if hasattr(args, 'segment_wikistep_sim_scores_DS_ready') and (
        not args.segment_wikistep_sim_scores_DS_ready):
        
        from datasets.build_knowledge.DS_get_sim_scores import DS_get_sim_scores
        DS_get_sim_scores(args, logger)

    if hasattr(args, 'segment_wikistep_sim_scores_ready') and (
        not args.segment_wikistep_sim_scores_ready):
        from datasets.build_knowledge.get_sim_scores import get_sim_scores
        get_sim_scores(args, logger)

    if hasattr(args, 'topics_formed') and not args.topics_formed:
        # TODO:
        from datasets.build_knowledge.get_topics import get_topics
        topic2task, task2topic = get_topics(args, logger)
    
    exit()
    if hasattr(args, 'nodes_formed') and not args.nodes_formed:
        
        from datasets.build_knowledge.get_nodes import get_nodes
        node2step, step2node = get_nodes(args, logger)
        
    if hasattr(args, 'edges_formed') and not args.edges_formed:
        
        if args.graph_structure == 'global':
            from datasets.build_knowledge.get_edges import get_edges
            pkg, G_wikihow, G_howto100m = get_edges(args, logger)
        elif args.graph_structure == 'local':
            from datasets.build_knowledge.get_edges_local_graph import get_edges
            pkg, G_wikihow, G_howto100m = get_edges(args, logger)
    exit()

    if hasattr(args, 'pseudo_label_DS_ready') and not args.pseudo_label_DS_ready:
        
        from datasets.build_knowledge.pseudo_label_DS import get_pseudo_label_DS
        get_pseudo_label_DS(args, logger)
        
    if hasattr(args, 'pseudo_label_VNM_ready') and not args.pseudo_label_VNM_ready:
        
        from datasets.build_knowledge.pseudo_label_VNM import get_pseudo_label_VNM
        get_pseudo_label_VNM(args, logger)
        
    if hasattr(args, 'pseudo_label_VTM_ready') and not args.pseudo_label_VTM_ready:
        
        from datasets.build_knowledge.pseudo_label_VTM import get_pseudo_label_VTM
        get_pseudo_label_VTM(args, logger)
        
    if hasattr(args, 'pseudo_label_TCL_ready') and not args.pseudo_label_TCL_ready:
       
        from datasets.build_knowledge.pseudo_label_TCL import get_pseudo_label_TCL
        get_pseudo_label_TCL(args, logger)
        
    if hasattr(args, 'pseudo_label_NRL_ready') and not args.pseudo_label_NRL_ready:
       
        from datasets.build_knowledge.pseudo_label_NRL import get_pseudo_label_NRL
        get_pseudo_label_NRL(args, logger)
        
    if hasattr(args, 'partition_dataset') and args.partition_dataset:
        
        from utils.dataset_utils import partition_dataset
        partition_dataset(args, logger)
       
    # os._exit(0)
    return




