import nfl_data_py as nfl
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC
import pandas as pd
from imblearn.combine import SMOTEENN

def GetSparkDF():
    '''

    A function to import the data and crate a Spark Data Frame for traning. This is to keep consistency across each model.

    Outputs:
        spark: The spark session
        undersampled_df3: The pyspark data frame to be used for testing
    
    '''

    years = []
    
    # loop through years from 2000 to 2023
    for i in range(2000,2024):
        years.append(i)
    
    # removing game_date, removing time
    # play_type is the predictor col
    cols = ["home_team", "away_team", "season_type", "week", "posteam", "posteam_type", 
            "defteam", "side_of_field", "yardline_100", "quarter_seconds_remaining", 
            "half_seconds_remaining", "game_seconds_remaining" , "game_half", "down", 
            "drive", "qtr",  "ydstogo", "play_type", "posteam_timeouts_remaining", 
            "defteam_timeouts_remaining", "posteam_score", "defteam_score", "score_differential", 
            "ep", "epa", "season"]
    
    #TODO add weather to col
    
    data = nfl.import_pbp_data(years, downcast=False, cache=False, alt_path=None)

    # get the desired columns
    reduced_data = data.filter(items=cols) 

    # select only where there are 4th downs
    forth_down = reduced_data.query("down==4.0")

    # set up the session
    spark = SparkSession.builder.getOrCreate()
    
    # Convert to PySpark DataFrame
    spark_df = spark.createDataFrame(forth_down)

    # remove nulls
    spark_df = spark_df.where(col("play_type").isNotNull())

    # removing QB kneel
    spark_df = spark_df.where(col("play_type") != "qb_kneel")

    # out sampling fractions
    fractions = {
        "field_goal": 0.6,
        "no_play": 0.6,
        "run": 0.6,
        "punt": 0.4,
        "pass": 0.6
    }
    
    # get the sample
    train_df = spark_df.sampleBy("play_type", fractions=fractions, seed=42)
    
    # remove sample to form the test
    test_df = spark_df.subtract(train_df)

    str_col = ["home_team", "away_team", "season_type", "posteam", "posteam_type", "defteam", "side_of_field", "game_half",
            "play_type", "season"]
    str_col_output = ["home_team_idx", "away_team_idx", "season_type_idx", "posteam_idx", "posteam_type_idx", "defteam_idx",
                      "side_of_field_idx", "game_half_idx", "play_type_idx", "season_idx"]
    ohe_col_input = ["home_team_idx", "away_team_idx", "season_type_idx", "posteam_idx", "posteam_type_idx", "defteam_idx",
                      "side_of_field_idx", "game_half_idx", "season_idx"]
    ohe_col_vec = ["home_team_vec", "away_team_vec", "season_type_vec", "posteam_vec", "posteam_type_vec", "defteam_vec",
                      "side_of_field_vec", "game_half_ivec", "season_vec"]

    # Convert PySpark DataFrame to pandas DataFrame
    pandas_df = train_df.toPandas()
    
    
    # Undersample the data
    rus = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    
    
    X_resampled, y_resampled = rus.fit_resample(pandas_df.drop("play_type", axis=1), pandas_df["play_type"])
    
    pandas_df.drop("play_type", axis=1, inplace=True)
    
    # Convert back to PySpark DataFrame
    undersampled_df = spark.createDataFrame(pd.DataFrame(X_resampled, columns=pandas_df.columns).assign(play_type=y_resampled))

    # Convert PySpark DataFrame to pandas DataFrame
    pandas_df = undersampled_df.toPandas()


    # Undersample the data
    rus2 = RandomUnderSampler(sampling_strategy='majority', random_state=42)
    
    
    X_resampled, y_resampled = rus2.fit_resample(pandas_df.drop("play_type", axis=1), pandas_df["play_type"])
    
    pandas_df.drop("play_type", axis=1, inplace=True)
    
    # Convert back to PySpark DataFrame
    undersampled_df2 = spark.createDataFrame(pd.DataFrame(X_resampled, columns=pandas_df.columns).assign(play_type=y_resampled))

    str_col2 = ["home_team", "away_team", "season_type", "posteam", "posteam_type", "defteam", "side_of_field", "game_half", "season"]
    
    # Convert PySpark DataFrame to pandas DataFrame
    pandas_df = undersampled_df2.toPandas()
    pandas_df2 = pandas_df.copy()
    pandas_df2.drop("play_type", axis=1, inplace=True)

    categorical_indices = [pandas_df2.columns.get_loc(col) for col in str_col2]

    # Define SMOTENC with indices of categorical columns
    smote_enn = SMOTENC(categorical_features=categorical_indices, random_state=42)
    #X_resampled, y_resampled = rus.fit_resample(pandas_df.drop("play_type", axis=1), pandas_df["play_type"])
    X_resampled, y_resampled = smote_enn.fit_resample(pandas_df.drop("play_type", axis=1), pandas_df["play_type"])
    pandas_df.drop("play_type", axis=1, inplace=True)
    
    # Convert back to PySpark DataFrame
    undersampled_df3 = spark.createDataFrame(pd.DataFrame(X_resampled, columns=pandas_df.columns).assign(play_type=y_resampled))

    return spark, undersampled_df3

    
    
    

    