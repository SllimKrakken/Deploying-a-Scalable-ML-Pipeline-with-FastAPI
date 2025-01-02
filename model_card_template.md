# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Travis Mills created this model on 01/02/2025. The model uses Logistics Regression which is created by Scikit-Learn library. This project is for the Machine Learning Devops class for WGU.
## Intended Use
The purpose is to predict the probability for people that earn over 50k annually from the U.S. Census data csv file.
## Training Data
The data was made up of the census data that was processed with one-hot encoding of categories. This was 80 percent of the data.
## Evaluation Data
The evaluation data was the remaining 20 percent. This also had one-hot encoding of categorical features as well.
## Metrics
_Please include the metrics used and your model's performance on those metrics._

The metrics used were Precision: 0.7466 | Recall: 0.6378 | F1: 0.6880. There are more metrics in the slice_output.txt file.
## Ethical Considerations
The data contains values for Genders and Race that can have some biases. Anyone who trains and handles the data should try and stay neutral on these areas when training the model. This can have an impact on the individuals not making over 50k annually based off those factors.
## Caveats and Recommendations
It would be recommended to use data for 2024 and not 2023. This would be good just so we can have current and up to date data. We want current data so it can keep up with inflation.

The model should be continually updated and evaluated each year.