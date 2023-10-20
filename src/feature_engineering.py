"""
feature_engineering.py
----------------------
DESCRIPCIÓN: "This is the Feature Engineering document for the
final project of the Machine Learning 2 course.
Here you will find the necessary functions for input data transformation.
Using best practices and object-oriented programming paradigm."

AUTOR: Lautaro Scheihing

FECHA: 14/10/2023

"""
#Import libraries
import logging
import argparse
import pandas as pd
import numpy as np

logging.basicConfig(
    filename='./logging_info_FE.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

class FeatureEngineeringPipeline(object):
    '''
    This class is used to perform the Feature Engineering stage of the project.
    '''
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path

    def read_data(self) -> pd.DataFrame:
        """
        This function reads the data from the path specified in the class constructor,
        and returns it as a Pandas DataFrame.

        :return pandas_df: The desired DataLake table as a DataFrame
        :rtype: pd.DataFrame
        """
        logging.info("Feature Engineering - Reading Data")
        # Import the data from the specified path
        pandas_df = pd.read_csv(self.input_path)
        #print(pandas_df.columns)
        # Return the DataFrame
        return pandas_df

    def data_transformation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Within this function, the data is transformed according to the needs of the project.
        We expect a Pandas DataFrame as input, and we return a Pandas DataFrame as output.
        The output should be a DataFrame with clean data, ready for the splitting stage.
        """

        logging.info("Feature Engineering - Transforming Data")
        #AGE OF THE OUTLET
        #First of all we make the transformation of the "Outlet Stablishment Year" to
        # "Outlet Age", we wont be changing the original column name, but the values inside it.
        #Assuming the year of the dataset creation is 2019,
        # we can substract the year of the establishment to get the age of the outlet.
        try:
            # Calculate the Outlet Age
            df['Outlet_Establishment_Year'] = 2019 - df['Outlet_Establishment_Year']
        except KeyError:
            print("Warning: 'Outlet_Establishment_Year' column not found.")


        #DROPPING COLUMNS
        #Now we could be changing the "Item Fat Content" column to a binary column,
        # where 0 is "Low Fat" and 1 is "Regular".
        # But taking into account this column serves no purpose for the model, we will drop it.
        try:
            # Drop unnecessary columns
            df = df.drop(columns=['Item_Fat_Content']).copy()
        except KeyError:
            print("Warning: 'Item_Fat_Content column not found for dropping not found.")

        #Another columns that wont be useful for the model are "Item Identifier"
        # and "Outlet_Identifier" so we will also be dropping those columns too.
        try:
            df = df.drop(columns=['Item_Type'])
        except KeyError:
            print("Warning: 'Item_Type' column not found.")

        #But before dropping the "Outlet_identifier" column we will use it to
        # fill the missing values of the "Outlet_Size" column.
        #First we will create a list of the unique values of the "Outlet_Identifier"
        # column that have a missing value in the "Outlet_Size" column.
        try:
            # Fill missing Outlet_Size values using Outlet_Identifier
            outlets = list(df[df['Outlet_Size'].isnull()]['Outlet_Identifier'].unique())
            #Then we iterate through the list of outlets and fill the missing
            # values with the mode of the "Outlet_Size" column of each outlet.
            for outlet in outlets:
                df.loc[df['Outlet_Identifier'] == outlet, 'Outlet_Size'] = 'Small'
            #With this step done, now we can drop the "Outlet_Identifier" column.
            df = df.drop(columns=['Outlet_Identifier'])
        except KeyError:
            print("Warning: One or more columns for filling missing values not found.")


        #ELIMINATING "Item_Weight" COLUMN.
        #As the "Item_Weight" column has a lot of missing values,
        # and we are pretty sure that it wont be useful for the model, we will drop it.
        try:
            # Drop 'Item_Weight'
            df = df.drop(columns=['Item_Weight'])
        except KeyError:
            print("Warning: 'Item_Weight column not found.")


        #ELIMINATION "Item_identifier" COLUMN.
        #As the "Item_identifier" column has a lot of unique values,
        # and we are pretty sure that it wont be useful for the model, we will drop it.
        try:
            df = df.drop(columns=['Item_Identifier'])
        except KeyError:
            print("Warning: 'Item_Identifier' column not found.")


        #ENCODING CATEGORICAL VARIABLES - Item_MRP
        #Now we will encode the "Item_MRP" column into 4 categories, based on 4
        # differen price ranges.
        try:
            #First we will create a list with the 4 price ranges. 0 to 100, 100 to 150,
            # 150 to 200 and 200 onwards.
            bins = [0, 100, 150, 200, np.inf]
            #Then we will create a list with the names of the categories.
            labels = [1, 2, 3, 4]
            #Finally we will create a new column with the encoded values.
            df['Item_MRP'] = pd.cut(df['Item_MRP'], bins=bins, labels=labels)
            #And we will change the type of the column to integer.
            df['Item_MRP'] = df['Item_MRP'].astype('int')
            #this step is done to avoid a warning message.
            #With this we obtain a new column with the encoded values of the "Item_MRP" column.
        except KeyError:
            print("Warning: 'Item_MRP' column not found.")


        #ENCODING CATEGORICAL VARIABLES - Outlet_Size and Outlet_Location_Type
        #Now we will encode the "Outlet_Size" and "Outlet_Location_Type" columns
        #into 3 categories each.
        try:
            df['Outlet_Size'] = df['Outlet_Size'].replace({'High': 2, 'Medium': 1, 'Small': 0})
        except KeyError:
            print("Warning: 'Outlet_Size' column not found.")
        try:
            df['Outlet_Location_Type'] = df['Outlet_Location_Type'].replace(
                {'Tier 1': 2, 'Tier 2': 1, 'Tier 3': 0})
        except KeyError:
            print("Warning: 'Outlet_Location_Type' not found.")


        #ENCODING CATEGORICAL VARIABLES - Outlet_Type
        #Now we will encode the "Outlet_Type" with the "get_dummies" function,
        # to create a new column for each category.
        # And as it is the last step of the data transformation, we will rename the
        # DataFrame and return it.
        try:
            # Encode 'Outlet_Type' using get_dummies
            df_transformed = pd.get_dummies(df, columns=['Outlet_Type'], dtype=int)
        except KeyError:
            print("Warning: 'Outlet_Type' column not found.")
        return df_transformed

    def write_prepared_data(self, transformed_dataframe: pd.DataFrame) -> None:
        """
        This function writes the transformed DataFrame to the path 
        specified in the class constructor.
        
        """
        logging.info("Feature Engineering - Writing Data")
        #We will write the DataFrame to a csv file, with the name
        # "data_transformed.csv" in the specified path.
        transformed_dataframe.to_csv(self.output_path + '_temp_data.csv', index=False)
        #We will print a message to confirm the file was written.
        print('The file was written successfully')


    def run(self):
        """
        This function runs the Feature Engineering Pipeline.
        """
        df = self.read_data()
        #return df
        df_transformed = self.data_transformation(df)
        self.write_prepared_data(df_transformed)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Feature Engineering")
    parser.add_argument('--input_path', type=str, help="Input path", required=True)
    parser.add_argument('--output_path', type=str, help="Output path", required=True)
    args = parser.parse_args()

    FeatureEngineeringPipeline(args.input_path, args.output_path).run()