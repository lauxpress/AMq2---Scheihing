"""
This module is the train pipeline
"""
import argparse
import subprocess

def feature_engineering(input_path, output_path):
    subprocess_args = ['python', 'feature_engineering.py',
                        '--input_path', input_path, 
                        '--output_path', output_path]
    subprocess.run(subprocess_args, check=True)

def train(input_path, model_path):
    subprocess_args = ['python', 'train.py',
                        '--input_path', input_path, 
                        '--model_path', model_path]
    subprocess.run(subprocess_args, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Pipeline")
    parser.add_argument('--input_path', type=str, help="Input path", required=True)
    parser.add_argument('--output_path', type=str, help="Output path", required=True)  # User-specified output path
    parser.add_argument('--model_path', type=str, help="Model path", required=True)
    args = parser.parse_args()

    # Call feature_engineering subprocess
    feature_engineering(args.input_path, args.output_path)

    # Call train subprocess with the output of feature_engineering as input
    train(args.output_path, args.model_path)