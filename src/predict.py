"""
predict.py


DESCRIPCIÓN: This script loads a trained model and makes predictions on a dataset.
AUTOR: Lautaro Scheihing
FECHA: 14/10/23
"""

# Imports
import pickle
import logging
import argparse
import os
import pandas as pd

logging.basicConfig(
    filename='./logging_info_pred.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


class MakePredictionPipeline(object):

    def __init__(self, input_path, model_path, pred_path: str = None):
        self.input_path = input_path
        self.model_path = model_path
        self.output_path = pred_path

    def load_data(self) -> pd.DataFrame:
        """
        This function loads the data from the specified path and return it as a Pandas DataFrame.
        """
        logging.info("Predict - Reading Data")
        # We import the data from the specified path
        pandas_df = pd.read_csv(self.input_path + '_temp_data.csv')

        return pandas_df

    def load_model(self) -> None:
        """
        This function will load the model from the specified path using the pickle library.
        """
        # We load the model from the specified path
        logging.info("Predict - Loading Model")
        self.model = pickle.load(open(self.model_path, 'rb'))


    def make_predictions(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This function will make predictions on the data using the model loaded in the previous function.
        """
        logging.info("Predict - Making Prediction")
        # We make the predictions
        new_data = data.copy()
        new_data['pred_Sales'] = self.model.predict(data)

        return new_data

    def write_predictions(self, predicted_data: pd.DataFrame) -> None:
        """
        This function will write the predictions to the specified path.
        """
        logging.info("Predict - Writing Prediction")
        # We write the predictions to the specified path

        #Specify the desired prediction file name (e.g., 'my_model.pkl')
        model_file_name = 'prediction.csv'

        # Combine the model path and file name to create the full file path
        file_path = os.path.join(self.output_path, model_file_name)

        # Check if the directory exists, and if it doesn't, create it
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        predicted_data.to_csv(file_path , index=False)


    def run(self):
        '''
        This function runs the pipeline.
        '''
        data = self.load_data()
        self.load_model()
        df_preds = self.make_predictions(data)
        self.write_predictions(df_preds)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predictions")
    parser.add_argument('--input_path', type=str, help="Input path (output from feature engineering)", required=True)
    parser.add_argument('--model_path', type=str, help="Model path", required=True)
    parser.add_argument('--pred_path', type=str, help="Output path for predictions", required=True)
    args = parser.parse_args()
    MakePredictionPipeline(args.input_path, args.model_path, args.pred_path).run()