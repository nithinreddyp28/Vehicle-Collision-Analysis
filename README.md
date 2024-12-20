Motor Vehicle Collisions Analysis - NYC

Introduction

This project analyzes motor vehicle collisions in New York City using a dataset collected by the NYPD. The dataset includes information such as crash date, time, borough, location, injuries, fatalities, contributing factors, and vehicle types. This analysis aims to support traffic safety initiatives like Vision Zero by providing insights into road safety and identifying patterns in traffic collisions.

Project Components

1. Data Cleaning

> The dataset contained several null and empty values. Key cleaning steps included:

> Assigning NA to empty cells and removing them as necessary.

> Dropping unwanted columns for a streamlined dataset.

> Adding a new column for fatalities to facilitate detailed analysis.

> Extracting and transforming date and year information for enhanced usability.

2. Exploratory Data Analysis (EDA): EDA was conducted to gain insights and uncover patterns in the dataset:

> Summary Statistics: Descriptive analysis to understand data distribution.

> Correlation Analysis: A correlation matrix revealed relationships between variables. For instance: persons_injured showed strong correlations with motorists_injured and weak correlations with persons_killed.

> Fatalities strongly correlated with persons_killed.

Visualization Highlights:

> Year-wise crash distribution showed 2021 had the most incidents.

> Bar plots highlighted top contributing factors and boroughs with high fatalities (e.g., Brooklyn and Queens).

3. Preliminary Analysis

> Multiple Linear Regression: Objective: Identify factors contributing to the number of injuries.

The best subset of predictors included:

> number_of_pedestrians_injured

> number_of_motorist_injured

> number_of_persons_killed

> number_of_cyclist_injured

Key Metrics:

> Adjusted R-squared: 95.3%

> Statistically significant predictors: pedestrians_injured, motorists_injured, and cyclists_injured.

> Regression diagnostics, including residual plots and VIF values, confirmed the model's validity.

4. Predictive Analysis: GLM Logistic Regression: Objective: Analyze factors contributing to fatalities in collisions.

> Binary outcome: Presence/absence of fatalities.

Results:

> Accuracy: 65.65%

> Precision: ~31.99%

> F1 Score: ~0.414

> AUC: 0.6465

> ROC Curve: Demonstrated moderate discriminatory ability.

Ridge and Lasso Regression: These models were applied to address multicollinearity and improve model performance. Significant predictors and regularization techniques enhanced the predictive framework.

5. Statistical Tests

ANOVA (Analysis of Variance)

> Examined differences in severity of injuries across contributing factors.

> Results showed significant effects for some contributing factors but not all.

Chi-Square Test for Independence

> Tested the association between borough and fatalities.

> Results indicated a significant relationship, highlighting spatial differences in traffic safety.

Key Insights: Top Contributing Factors:

> Distraction

> Right-of-way violations

> Unsafe speed

> High-Risk Areas: Brooklyn and Queens exhibited the highest fatalities.

Modeling:

> Regression models provided actionable insights for predicting injuries and fatalities.

> Statistical tests validated the significance of key variables.

Conclusion

This analysis underscores the importance of targeted traffic safety measures and highlights key factors contributing to injuries and fatalities in NYC motor vehicle collisions. The findings support data-driven policy decisions and ongoing initiatives like Vision Zero.
