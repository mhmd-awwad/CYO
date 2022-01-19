#############################################################
# Create Data set
#############################################################

# Install and Load Packages
if(!require(plotly)) install.packages("plotly", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("corrplot", repos = "http://cran.us.r-project.org")
if(!require(RCurl)) install.packages("RCurl", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")

library(dplyr)
library(ggplot2)
library(caret)
library(tidyr)
library(plotly)
library(RCurl)
library(corrplot)
library(randomForest)
library(kableExtra)

# Video Game Sales with Ratings
# Source File: https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings
# File Download Path: https://github.com/mhmd-awwad/CYO/raw/main/Video_Games_Sales_as_at_22_Dec_2016.csv"

URL <- tempfile()
download.file("https://github.com/mhmd-awwad/CYO/raw/main/Video_Games_Sales_as_at_22_Dec_2016.csv",URL)

rawdata <- read.csv(file=URL)

# Raw Data Checking: Columns
head(rawdata)

# Raw Data Checking: Type of each Column
str(rawdata)

# Raw Data Checking: Statistic of each colum
summary(rawdata)

# Data Cleansing: Remove invalid records of "Year of Release" marked "NA"
cleandata <- rawdata %>% filter(!is.na(rawdata$Year_of_Release))

# Data generated in Nov 2016
# Data Cleansing: Change "Year of Release" to numeric and Remove invalid records of "Year of Release" after 2016
cleandata <- cleandata%>% dplyr::filter((as.numeric(as.character(cleandata$Year_of_Release)))<=2016)

# Data Cleansing: Correct record with wrong "Rating"
cleandata_rp <- cleandata %>% filter(Rating=="RP")
cleandata_rp
cleandata$Rating[cleandata$Rating == 'RP'] <- "E10+"

# Data Cleansing: Remove invalid records of game with blank in "name"
cleandata <- cleandata %>% filter(cleandata$Name!="")

# Data Cleansing: Change "User_Score" to numeric
cleandata$User_Score <- as.numeric(as.character(cleandata$User_Score))

# Data Cleansing: Remove NA rows
finaldata <- na.omit(cleandata)

# Data Exploration: No. of rows and columns final dataset
dim(finaldata)
finaldata_record<-nrow(finaldata)

# Data Exploration: Statistic of each colum of final dataset
summary(finaldata)

# Data Exploration: No. of video games in final dataset
n_distinct(finaldata$Name)
game_no<-n_distinct(finaldata$Name)

# Data Exploration: No. of Critic ratings in final dataset
finaldata_genres <- finaldata %>% group_by(Genre) %>%
  summarise(Critic_Rating=sum(Critic_Count)) %>%
  arrange(desc(Critic_Rating))

# Data Exploration: No. of Ratings by Genres Plot
finaldata_genres_p <-finaldata_genres%>%plot_ly(
  x = finaldata_genres$Genre,
  y = finaldata_genres$Critic_Rating,
  name = "Rating Distribution by Genres",
  type = "bar"
) %>% 
  add_text(text=finaldata_genres$Critic_Rating, hoverinfo='none', textposition = 'top', showlegend = FALSE,
           textfont=list(size=10, color="black"))%>%
  layout(xaxis = list(title = "Genres"),
         yaxis = list(title = "No. of Rating"))
finaldata_genres_p

# Data Exploration: Top 10 video game with the greatest No. of Critic ratings
finaldata_rating <- finaldata %>% group_by(Name) %>%
  summarize(Critic_Rating=sum(Critic_Count)) %>%
  top_n(10) %>%
  arrange(desc(Critic_Rating))
kable(finaldata_rating) %>%
  kable_styling(full_width = F) %>%
  column_spec(1, width = "20em")

# Data Exploration: Sales in North America vs Critic Scores
finaldata_NAsales <- finaldata %>%
  group_by(Critic_Score) %>%
  summarize(NA_Sales=sum(NA_Sales)) 

# Data Exploration: Sales in North America vs Critic Scores Plot
finaldata_NAsales_p <- plot_ly(finaldata_NAsales, x = ~finaldata_NAsales$Critic_Score, y = ~finaldata_NAsales$NA_Sales, type = 'scatter', mode = 'lines')%>%
  layout(xaxis = list(title = "Metascore"),
         yaxis = list(title = "Sales in North America (in millions of units)"))
finaldata_NAsales_p



#############################################################
# Create a train set and test set from finaldata dataset
#############################################################

# test set will be 20% of final dataset
set.seed(1)
NASales_test_index <- createDataPartition(y = finaldata$NA_Sales, times = 1, p = 0.2, list = FALSE)
NASales_train_set <- finaldata[-NASales_test_index,]
NASales_test_set <- finaldata[NASales_test_index,]



#############################################################
# Define Residual Mean Squared Error (RMSE) 
#############################################################

RMSE <- function(true_NA_Sales, predicted_NA_Sales){
  sqrt(mean((true_NA_Sales - predicted_NA_Sales)^2))
}



#############################################################
# Model: Linear Regression
#############################################################

# Build the model on train dataset
lmModel <- lm(NA_Sales ~ Critic_Score, data=NASales_train_set) 

# Predict test dataset
lmPred <- predict(lmModel, NASales_test_set)  

# Model prediction performance
lm_rmse <- RMSE(lmPred, NASales_test_set$NA_Sales)
lm_rmse

# Create a Results Table
rmse_results <- data_frame(method = "Linear Regression", RMSE = lm_rmse)
rmse_results 


#############################################################
# Model: Polynomial Regression
#############################################################

# Build the model on train dataset
polyModel <- lm(NA_Sales ~ Critic_Score+ I(Critic_Score^2) + I(Critic_Score^3), data=NASales_train_set) 

# Predict test dataset
polyPred <- predict(polyModel, NASales_test_set)  

# Model prediction performance
poly_rmse <- RMSE(polyPred, NASales_test_set$NA_Sales)
poly_rmse

# Add Polynomial Regression result to the Results Table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Polynomial Regression",  
                                     RMSE = poly_rmse))



#############################################################
# Model: Elastic Net
#############################################################

# Build the model on train dataset
enModel <- train(
  NA_Sales~Critic_Score+ I(Critic_Score^2) + I(Critic_Score^3), data = NASales_train_set, method = "glmnet",
  trControl = trainControl("cv", number = 10),
  tuneLength = 10
)

# Model coefficients
coef(enModel$finalModel, enModel$bestTune$lambda)

# Make predictions
enPred<- enModel %>% predict(NASales_test_set)

# Model prediction performance
en_rmse <- RMSE(enPred, NASales_test_set$NA_Sales)
en_rmse

# Add Elastic net result to the Results Table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Elastic Net",  
                                     RMSE = en_rmse))



#############################################################
# Model: Random Forest
#############################################################

# Build the model on train dataset
rfModel <- randomForest(NA_Sales ~ Critic_Score, data = NASales_train_set, importance = TRUE)

# Predict test dataset
rfPred <- predict(rfModel, NASales_test_set)  

# Model prediction performance
rf_rmse <- RMSE(rfPred, NASales_test_set$NA_Sales)
rf_rmse 

# Add Random Forest result to the Results Table
rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Random Forest",  
                                     RMSE = rf_rmse))
rmse_results
kable(rmse_results) %>%
  kable_styling(full_width = F) %>%
  column_spec(1, width = "20em")
