# CS488_Project
Stock Market analysis and prediction pipeline using sentiment analysis on different sections of The Guardian Newspaper. 


---
![Architecture Diagram](software_architecture.png?raw=true "Architecture Diagram")



## Demo

Instructions for running are placed in demo.ipynb along with the output for a working version for the year 2010. 


## Functionalities

*compile_articles.py-- takes start and end date to connect to Guardian API and downloads data. 
*compile_dataframes.py -- Uses the articles collected from Compile_articles to create a combined dataframe for processing.
*compile_stock_data.py --  Connects to Yahoo finance API to collect data for 5 stock indices for a given timeframe.
*compile_sentiment_scores.py -- Uses dataframe from Compile_dataframe to generate sentiment scores. 
*visualize_and_predict.py --- Visualizes and predicts results. 


![Data Collection](data_collection.png?raw=true "Data Collection Method")