# CS488_Project

Stock Market Trend Prediction Using Sentiment Analysis


Instructions for running are placed in demo.py along with the output for a working version for the year 2010. 


Compile_articles-- takes start and end date to connect to Guardian API and downloads data 
Compile_dataframes -- Uses the articles collected from Compile_articles to create a combined dataframe for processing
Compile_stock_data --  Connects to Yahoo finance API to collect data for 5 stock indices for a given timeframe
Compile_sentiment_scores -- Uses dataframe from Compile_dataframe to generate sentiment scores 
Visualize_and_predict --- Visualizes and predicts results # CS488_Project
