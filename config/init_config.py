import argparse
import json
import os

from flashrag.config import Config

def init_cfg():
    # Create argument parser
    parser = argparse.ArgumentParser("Update configuration from command line arguments.")

    parser.add_argument("--method_name", type=str, help="Name of the method",
                        default="naive_rag")
    parser.add_argument("--config_path", type=str, help="Path to the config.json file")
    
    # Global Paths
    parser.add_argument("--model_path", type=str, help="Path to the pre-trained model")
    parser.add_argument("--generator_model", type=str, help="Name of the generator model",
                        default="llama3-8B-instruct")
    
    # Environment Settings   
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset",
                        choices=["StrategyQA", "HotpotQA", "NaturalQuestions"])
    parser.add_argument("--test_sample_num", type=int, help="Number of questions",
                        default=100)
    parser.add_argument("--save_dir", type=str, help="Output directory to save the responses",
                        default="output/")

    # Debate arguments
    parser.add_argument("--agents", type=int, help="Number of agents",
                        default=2)
    parser.add_argument("--rag_agents", type=int, help="Number of RAG agents",
                        default=1)
    parser.add_argument("--rounds", type=int, help="Number of rounds",
                        default=5)

    parser.add_argument("--gpu_id", type=str, default="0")
    
    args = parser.parse_args()
    
    if args.method_name is None:
        raise ValueError("Please choose a method")
    
    if args.config_path is None:
    # warn if config file not found
        print("Warning: No config file provided. Using base configuration.")
        config = {}
        
    else:
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(f"Config file not found at {args.config_path}")
        
        with open(args.config_path, "r") as f:
            config = json.load(f)

    # Step 4: Update config dictionary with provided arguments
    for key, value in vars(args).items():
        if key != "config_path" and value is not None:  # Skip config_path and only update provided arguments
            config[key] = value
    print(config)

    config["save_note"] = args.method_name
    config["save_dir"] = os.path.join(config["save_dir"], config["generator_model"], config["save_note"])
    
    config =  Config("./config/base_config.yaml", config)
    
    return config
