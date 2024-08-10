# Reveal-Categories-Found-in-Data
Apply clustering and NLP techniques to sort text data from negative reviews in the Google Play Store into categories.

# Description: 
App publishers want to know what users think and feel about their apps. While we can't read their minds, you can data-mine the reviews they leave to uncover the main topics!

In this project, you will use clustering and NLP techniques to sort text data from negative reviews in the Google Play Store into categories.

![image](https://github.com/user-attachments/assets/35998a4b-27c1-4fbe-afec-1025df7e734b)



As a Data Scientist working for a mobile app company, you usually find yourself applying product analytics to better understand user behavior, uncover patterns, and reveal insights to identify the great and not-so-great features. Recently, the number of negative reviews has increased on Google Play, and as a consequence, the app's rating has been decreasing. The team has requested you to analyze the situation and make sense of the negative reviews.

It's up to you to apply K-means clustering from scikit-learn and NLP techniques through NLTK to sort text data from negative reviews in the Google Play Store into categories!

## The Data

A dataset has been shared with a sample of reviews and their respective scores (from 1 to 5) in the Google Play Store. A summary and preview are provided below.

# reviews.csv

| Column     | Description              |
|------------|--------------------------|
| `'content'` | Content (text) of each review. |
| `'score'` | Score assigned to the review by the user as an integer (from 1 to 5). |
