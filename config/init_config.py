import argparse
import json

def parse_args():
    # Create argument parser
    parser = argparse.ArgumentParser()

    # Add argument to specify config file
    parser.add_argument("--config", type=str, help="Path to the config.json file",
                        default="./config/Debate_Llama-8B.json")

    # Parse the initial arguments to get the config file path
    args, remaining_args = parser.parse_known_args()

    # Load configurations if config file is provided
    config = {}
    if args.config:
        try:
            with open(args.config, "r") as config_file:
                config = json.load(config_file)
        except FileNotFoundError:
            print(f"Warning: Config file '{args.config}' not found. Using default parameters.")

    # Define arguments with defaults from the config file
    parser.add_argument("--dataset", type=str, help="Name of the dataset",
                        default=config.get("dataset", "MMLU"),
                        choices=["MMLU", "StrategyQA", "HotpotQA"])
    parser.add_argument("--model", type=str, help="Name of the model",
                        default=config.get("model", "Llama-3.1-8B"))
    parser.add_argument("--model_id", type=str, help="Path to the pre-trained model",
                        default=config.get("model_id", "/data/share/llama_checkpoints/Llama-3.1-8B-Instruct"))
    parser.add_argument("--output_dir", type=str, help="Output directory to save the responses",
                        default=config.get("output_dir", "./outputs"))
    
    parser.add_argument("--paradigm", type=str, help="Paradigm of the method",
                        default=config.get("paradigm", "DRAG"),
                        choices=["DRAG", "single_agent", "cot", "reflection", "RAG"])
    parser.add_argument("--use_summary_agent", type=bool, help="Whether to use the summary agent",
                        default=config.get("use_summary_agent", True))

    parser.add_argument("--agents", type=int, help="Number of agents",
                        default=config.get("agents", 2))
    parser.add_argument("--rag_agents", type=int, help="Number of RAG agents",
                        default=config.get("rag_agents", 1))
    parser.add_argument("--rounds", type=int, help="Number of rounds",
                        default=config.get("rounds", 3))
    parser.add_argument("--question_cnts", type=int, help="Number of questions",
                        default=config.get("question_cnts", 100))
    parser.add_argument("--top_k_results", type=int, help="Number of top k results to retrieve",
                        default=config.get("top_k_results", 3))
    parser.add_argument("--doc_content_chars_max", type=int, help="Maximum number of characters in the document content",
                        default=config.get("doc_content_chars_max", 1000))
    parser.add_argument("--rewrite_query", type=bool, help="Whether to rewrite the query for RAG agents",
                        default=config.get("rewrite_query", True))
    parser.add_argument("--device", type=str, default=config.get("device", "cuda:0"))
    parser.add_argument("--max_new_tokens", type=int, help="Maximum number of new tokens to generate",
                        default=config.get("max_new_tokens", 512))

    parser.add_argument("--seed", type=int, help="Random seed",
                        default=config.get("seed", 42))

    # Parse all arguments, including any overrides
    return parser.parse_args(remaining_args)
