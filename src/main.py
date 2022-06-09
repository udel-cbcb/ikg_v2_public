import argparse
from loguru import logger
import logging
import os

#logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--command', metavar='--c')
args = parser.parse_args()
command = args.command

################ DATA #################
if command == "build_kg":
    import data.kg_builder as kgb
    kgb.build_kg()

elif command == "create_viz_subset":
    import data.visualizer as vz
    vz.create_viz_subset()

############### NEGATIVE SAMPLING DATA ###################################
elif command == "build_negative_sampling_exp_data":
    os.system("python -m src.data.negative_sampling.data_builder")

elif command == "build_protein_go_embedding":
    os.system("python -m src.data.negative_sampling.protein_go_embedder")

elif command == "build_negative_sampling_edge_data":
    os.system("python -m src.data.negative_sampling.edge_builder")

############### HETEOGENOUS EMBEDDING DATA ###################################
elif command == "build_hetero_embedding_data":
    os.system("python -m src.data.hetero_embedding.data_builder")

elif command == "build_ec_data":
    os.system("python -m src.data.hetero_embedding.ec_builder")

elif command == "build_hetero_embedding_index":
    os.system("python -m src.data.hetero_embedding.build_index")

elif command == "build_hetero_embedding_protein_to_go_embedding":
    os.system("python -m src.data.hetero_embedding.protein_go_embedder")

elif command == "build_hetero_embedding_negative_edges":
    os.system("python -m src.data.hetero_embedding.build_negative_edges")

############### FEATURE ANALYSIS DATA ###################################
elif command == "build_feature_analysis_data":
    os.system("python -m src.data.feature_analysis.data_builder")

elif command == "build_feature_analysis_index":
    os.system("python -m src.data.feature_analysis.build_index")

elif command == "build_mesh_to_doid_mapping":
    os.system("python -m src.data.feature_analysis.build_mesh_to_doid")

############### ABALATION STUDY DATA ###################################
elif command == "build_abalation_study_data":
    os.system("python -m src.data.abalation_study.data_builder")

elif command == "build_abalation_study_index":
    os.system("python -m src.data.abalation_study.build_index")

############### BIOINFORMATIC ANALYSIS DATA ###################################
elif command == "generate_kinase_list":
    os.system("python -m src.data.bioinformatic_analysis.generate_kinase_list")

elif command == "convert_uniprot_id":
    os.system("python -m src.data.bioinformatic_analysis.convert_uniprot_id")

elif command == "make_predictions":
    os.system("python -m src.data.bioinformatic_analysis.make_predictions")

elif command == "generate_analysis_candidates":
    os.system("python -m src.data.bioinformatic_analysis.generate_analysis_candidates")

elif command == "build_test_edges_compare":
    os.system("python -m src.data.bioinformatic_analysis.build_test_edges_compare")

################ EXPERIMENT ###########
elif command == "negative_sampler_compare":
    os.system("python -m src.experiment.negative_sampler_compare")

elif command == "hetero_embedding_compare":
    os.system("python -m src.experiment.hetero_embedding_compare")

elif command == "feature_analysis_compare":
    os.system("python -m src.experiment.feature_analysis_compare")

################ TEST ###########
elif command == "kg_embedder_test":
    from embedder import kg_embedder
    kg_embedder.test()

elif command == "node2vec_embedder_test":
    from model_tests import node2vec_embedder_test
    node2vec_embedder_test.test()

elif command == "node2vec_embedder_test_geo":
    from model_tests import node2vec_geometric_test
    node2vec_geometric_test.test()

elif command == "node2vec_embedder_test_ikg":
    from embedder import node2vec_embedder
    node2vec_embedder.test()

elif command == "train_rf_model_test":
    from model import rf_model
    rf_model.test()

elif command == "train_prediction_test":
    from train import train_prediction
    train_prediction.test()

else:
    logger.error(f"Unknown command : {command}")

