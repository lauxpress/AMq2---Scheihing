"""
train.py

COMPLETAR DOCSTRING

DESCRIPCIÓN: This script is used to train the model. It takes the data from the Feature 
Engineering stage, trains the model and dumps it into a pickle file.
AUTOR: Lautaro Scheihing
FECHA: 14/10/23
"""
# Imports
import logging
import argparse
import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename='./logging_info_train.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

class ModelTrainingPipeline(object):
    '''
    This class is used to perform the Model Training stage of the project.
    '''
    def __init__(self, input_path, model_path):
        self.input_path = input_path
        self.model_path = model_path

    def read_data(self) -> pd.DataFrame:
        """
        This function reads the data from the path specified in the class constructor, 
        and returns it as a Pandas DataFrame.
        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
        logging.info("Train - Reading Data")
        #We import the data from the specified path
        pandas_df = pd.read_csv(self.input_path + '_temp_data.csv')
        return pandas_df

    def model_training(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        This function takes the DataFrame from the previous function and trains a model with it.
        """
        logging.info("Train - Training Model")
        #We create the model
        model = LinearRegression()
        seed = 28
        #We split the data into train and test
        X = df.drop(columns='Item_Outlet_Sales')
        x_train, x_val, y_train, y_val = train_test_split(X, df['Item_Outlet_Sales'],
                                                          test_size = 0.3, random_state=seed)

        #We train the model
        model.fit(x_train, y_train)
        #We predict the values for the validation set
        pred = model.predict(x_val)

        #Now we calculate the mse and the R^2 coefficient for the train and validation sets
        mse_train = metrics.mean_squared_error(y_train, model.predict(x_train))
        R2_train = model.score(x_train, y_train)
        print('Model Metrics:')
        print('Training: RMSE: {:.2f} - R2: {:.4f}'.format(mse_train**0.5, R2_train))

        mse_val = metrics.mean_squared_error(y_val, pred)
        R2_val = model.score(x_val, y_val)
        print('VALIDACIÓN: RMSE: {:.2f} - R2: {:.4f}'.format(mse_val**0.5, R2_val))

        return model

    def model_dump(self, model_trained) -> None:
        """
        We now dump the model into a pickle file.
        """
        logging.info("Train - Dumping Model")
        #We use pickle to dump the model into a pickle file so that
        # we can later use it to make predictions
        #Specify the desired model file name (e.g., 'my_model.pkl')
        model_file_name = 'model.pkl'

        # Combine the model path and file name to create the full file path
        file_path = os.path.join(self.model_path, model_file_name)

        # Check if the directory exists, and if it doesn't, create it
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        # Save the model to the specified file
        with open(file_path, 'wb') as file:
            pickle.dump(model_trained, file)
        
        print('The model was saved successfully')


    def run(self):
        '''
        This function runs the whole pipeline.
        '''
        df = self.read_data()
        model_trained = self.model_training(df)
        self.model_dump(model_trained)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--input_path', type=str, help="Input path", required=True)
    parser.add_argument('--model_path', type=str, help="Model path", required=True)
    args = parser.parse_args()

    ModelTrainingPipeline(args.input_path, args.model_path).run()