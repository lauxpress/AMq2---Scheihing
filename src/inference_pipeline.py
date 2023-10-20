"""
This module is the inference pipeline
"""
import argparse
import os
import subprocess

def feature_engineering(input_path, output_path):
    subprocess_args = ['python', 'feature_engineering.py',
                        '--input_path', input_path, 
                        '--output_path', output_path]
    subprocess.run(subprocess_args, check=True)

def predict(input_path, model_path, pred_path):
    subprocess_args = ['python', 'predict.py',
                        '--input_path', input_path,  # Use the full path
                        '--model_path', model_path,
                        '--pred_path', pred_path]
    subprocess.run(subprocess_args, check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Pipeline")
    parser.add_argument('--input_path', type=str, help="Input path", required=True)
    parser.add_argument('--output_path', type=str, help="Output path", required=True)  # User-specified output path for feature engineering
    parser.add_argument('--model_path', type=str, help="Model path", required=True)
    parser.add_argument('--pred_path', type=str, help="Prediction output path", required=True)
    args = parser.parse_args()

    # Call feature_engineering subprocess
    feature_engineering(args.input_path, args.output_path)

    # Call predict subprocess with the full path to the feature engineering output
    predict(args.output_path, args.model_path, args.pred_path)

    print(f"Predictions saved to {args.pred_path}")