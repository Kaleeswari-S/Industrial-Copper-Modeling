# Industrial-Copper-Modeling
## Introduction:

The Industrial Copper Modeling project focuses on predicting the selling price and status (won or lost) in the industrial copper market using machine learning regression and classification algorithms. By exploring the dataset, performing data cleaning and preprocessing, and applying various machine learning techniques, we aim to develop models that can accurately predict the selling price and status in the copper market.

## Dataset:

The dataset used for this analysis contains information about industrial copper transactions, including variables such as selling price, quantities, and status (won or lost). It provides a comprehensive view of the copper market and factors that influence the outcomes of transactions.

[Data](https://docs.google.com/spreadsheets/d/18eR6DBe5TMWU9FnIewaGtsepDbV4BOyr/edit?usp=sharing&ouid=104970222914596366601&rtpof=true&sd=true)

## About the Data:

1. `id`: This column likely serves as a unique identifier for each transaction or item, which can be useful for tracking and record-keeping.

2. `item_date`: This column represents the date when each transaction or item was recorded or occurred. It's important for tracking the timing of business activities.

3. `quantity tons`: This column indicates the quantity of the item in tons, which is essential for inventory management and understanding the volume of products sold or produced.

4. `customer`: The "customer" column refers to the name or identifier of the customer who either purchased or ordered the items. It's crucial for maintaining customer relationships and tracking sales.

5. `country`: The "country" column specifies the country associated with each customer. This information can be useful for understanding the geographic distribution of customers and may have implications for logistics and international sales.

6. `status`: The "status" column likely describes the current status of the transaction or item. This information can be used to track the progress of orders or transactions, such as "Draft" or "Won."

7. `item type`: This column categorizes the type or category of the items being sold or produced. Understanding item types is essential for inventory categorization and business reporting.

8. `application`: The "application" column defines the specific use or application of the items. This information can help tailor marketing and product development efforts.

9. `thickness`: The "thickness" column provides details about the thickness of the items. It's critical when dealing with materials where thickness is a significant factor, such as metals or construction materials.

10. `width`: The "width" column specifies the width of the items. It's important for understanding the size and dimensions of the products.

11. `material_ref`: This column appears to be a reference or identifier for the material used in the items. It's essential for tracking the source or composition of the products.

12. `product_ref`: The "product_ref" column seems to be a reference or identifier for the specific product. This information is useful for identifying and cataloging products in a standardized way.

13. `delivery date`: This column records the expected or actual delivery date for each item or transaction. It's crucial for managing logistics and ensuring timely delivery to customers.

14. `selling_price`: The "selling_price" column represents the price at which the items are sold. This is a critical factor for revenue generation and profitability analysis.

## Regression model details:

The copper industry deals with less complex data related to sales and pricing. However, this data may suffer from issues such as skewness and noisy data, which can affect the accuracy of manual predictions. Dealing with these challenges manually can be time-consuming and may not result in optimal pricing decisions. A machine learning regression model can address these issues by utilizing advanced techniques such as data normalization, outlier detection and handling, handling data in wrong format, identifying the distribution of features, and leveraging tree-based models, specifically the decision tree algorithm.

## Classification model details:

Another area where the copper industry faces challenges is in capturing the leads. A lead classification model is a system for evaluating and classifying leads based on how likely they are to become a customer. You can use the STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.

## Learn from the Project:

Exploring Skewness and Outliers: Analyze the distribution of variables in the dataset and identify skewness and outliers. This step helps in understanding the data quality and potential issues that may affect the model performance.

Data Transformation and Cleaning: Transform the data into a suitable format for analysis and perform necessary cleaning steps. This includes handling missing values, encoding categorical variables, and scaling numerical features.

Machine Learning Regression Algorithms: Apply various machine learning regression algorithms to predict the selling price of industrial copper. Compare the performance of algorithms such as linear regression, decision trees, random forests, or gradient boosting.

Machine Learning Classification Algorithms: Apply different machine learning classification algorithms to predict the status (won or lost) of copper transactions. Explore algorithms such as logistic regression, support vector machines, or random forests to classify the outcomes.

Evaluation and Model Selection: Evaluate the performance of regression and classification models using appropriate metrics such as    mean squared error (MSE), accuracy, precision, and recall. Select the best-performing models based on these metrics.

## Methodology:

### 1.Data Loading:
Load the industrial copper dataset into the code using pandas library. Perform initial data exploration to understand the structure and content of the dataset.

### 2.Data Cleaning and Preprocessing:
Handle missing values, remove outliers if necessary, and perform necessary data transformations such as encoding categorical variables. This step ensures the data is in a suitable format for analysis.

### 3.Exploratory Data Analysis (EDA):
Use pandas, matplotlib, and seaborn libraries to explore the dataset. Analyze different variables, their distributions, and relationships. Generate visualizations such as histograms, scatter plots, or box plots to gain insights into the data.

### 4.Machine Learning Regression:
Apply various machine learning regression algorithms to predict the selling price of industrial copper. Split the dataset into training and testing sets, train the models, and evaluate their performance using metrics such as mean squared error (MSE).

### 5.Machine Learning Classification:
Apply different machine learning classification algorithms to predict the status (won or lost) of copper transactions. Split the dataset into training and testing sets, train the models, and evaluate their performance using metrics such as accuracy, precision, and recall.

### 6.Documentation:
Prepare a comprehensive documentation summarizing the steps involved in the analysis, including the preprocessing techniques, machine learning algorithms used, and their performance. Include visualizations and interpretations to effectively communicate the results.

## Conclusion:
The Industrial Copper Modeling project aims to predict the selling price and status in the industrial copper market using machine learning techniques

### Project Demo Video: 

## Contributing
Contributions to this project are welcome! If you encounter any issues or have suggestions for improvements, please feel free to submit a pull request.

## Contact:

📧 Email: kaleeswariramkumar25@gmail.com 

🌐 LinkedIn: [linkedin.com/in/kaleeswari-s](https://www.linkedin.com/in/kaleeswari-s-081a392a6/)

For any further questions or inquiries, feel free to reach out. We are happy to assist you with any queries.

