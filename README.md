# **Keeping Netflix on Top** 
### *Analyzing User Reviews to Prevent Customer Churn & Attract New Subscribers*
- **Author:** Mohammad Abou-Ghazala
- **Date:** March, 2025

![image](https://github.com/momopajamas/predicting_netflix_user_reviews/blob/main/images/netflix_logo.png?raw=true)

# Introduction
### Current State of Streaming
Though Netflix has long dominated the OTT market ("Over-the-Top", refering to the method of delivering content over the internet), the landscape of streaming services has drastically changed over the last few years, and with it, the challenges that Netflix faces. One of the most formidable of these challenges is the entry of legacy media companies, such as Disney and NBCUniversal, into the OTT market, as they are able to [leverage their deep content libraries to attract subscribers to their own services](https://www.filmtake.com/streaming/chasing-netflix-how-the-major-media-companies-stack-up-in-subscribers-revenue-and-challenges-part-one/).

According to [*Whip Media's 2023 US Streaming Satisfaction Survey*](https://whipmedia.com/wp-content/uploads/2023/09/2023-US-Streaming-Satisfaction-Study.pdf), there is persistent weakening trend among platform leaders, as Netflix "continues to decline in most measurements of satisfaction, while still holding on to the top rankings in the categories of user experience and programming recommendations."

A [*2023 piece by Variety surveying 40 Hollywood insiders*](https://variety.com/lists/user-friendly-streaming-services-survey/) about their experience using the numerous streaming platforms available to the public, or more specifically, "the way consumers interact with an app or website before, during and after they watch that content", found that "Netflix was most widely regarded as having the best UI [User Interface], with several comments about how intuitive and aesthetically pleasing the platform is." Another strength of Netflix's is its sophisticated recommendation system, sometimes refered to as ["digital nudging", which plays an important role in minimizing customer churn](https://journals.sagepub.com/doi/10.1177/20438869241296895?icid=int.sj-full-text.citing-articles.1&utm_source=chatgpt.com).

Another aspect that gives Netflix a competitive advantage over other streaming platforms is [the state of its global reach, as it currently operates in more than 190 countries](https://canvasbusinessmodel.com/blogs/competitors/netflix-competitive-landscape#:~:text=Global%20Reach:%20Netflix%20operates%20in,content%2C%20known%20as%20Netflix%20Originals.).

While other streaming services can leverage content exclusivity to attract subscribers, Netflix's library is not as deep or expansive, at least for the time being. However, Netflix's technical superiorty in UI and digital nudging gives it an indispensible competitive edge against its competitors, and must continually adjust its platforms and respond to the faults which can drive away consumers and subscribers.

# Business Understanding
Considering the strengths and weaknesses of Netflix outlined above, we need maintain our competitive edge against other streaming platforms by better understanding which aspects of our platform **frustrate our customers** and which aspects **resonate well with subscribers**.

In other words, both **Positive ad Negative reviews matter equally for our purposes**.

We can do so by taking a look at reviews written over the past two years (2023-2025), to ensure that our insights are timely and relevant, as there is little point in working with older criticisms that may have already been addressed. 

Put another way, we seek insights pertaining to the following:
1. **Customer Retention**, and why some users are unhappy with Netflix and may cancel their subscriptions.
2. **Customer Acquisition**, and identify what it is that people currently appreciate about Netflix so it can inform our promotions and marketing.

What this requires of us is:

1. To build a **binary classification model** that can deploy Natural Language Processing (NLP), or more specifically, Sentiment Analysis on the user reviews, and use that insight to predict whether the review is positive or negative. This can be done through traditional supervised learning models, such as Logistic Regression, Random Forest Classifiers, SVM, etc.
2. **Apply Latent Dirichlet Allocation**, or `LDA`, to both groups of positive and negative reviews to identify trends and themes that will inform the actions we take.

In this way, we will be better equipped to provide solid business recommendations regarding subscriber retention as well as more effective marketing campaigns.

# Data Understanding
The [dataset we will be working with was pulled from Kaggle](https://www.kaggle.com/datasets/ashishkumarak/netflix-reviews-playstore-daily-updated?resource=download) and contains more than 129,000 reviews dating back to 2018, which is 7 years as of the time of writing. This dataset is updated daily, and the data contained within it is up-to-date as of 2 March, 2025.

Of the 8 columns contained in this dataset, the following are of particular or potential interest to us, pending further investigation:

1. `content`, which contains the text of the user review. We will use this column for our NLP and Sentiment Analysis.
2. `# score`, which contains a discrete (categorical) numeric rating on a scale of 1-5. This will serve as our Target column whose Labels we will predict based on the the text of the user reviews.
3. `# thumbsUpCount`, which tells us how many 'thumbs up' each user review received from other reviews, potentially indicating the relative significance of different reviews since a higher count of thumbs up would indicate the review resonated with other users.
4. `at`, telling us the date the user review was created. We will be looking to filter our data so that we can focus on reviews produced over the past year 1-2 years, or since 2023.

### Features
We will use the text contained within the `content` column to produce features for our classifier by vectorizing the text using **Term Frequency-Inverse Document Frequency** (TF-IDF), which is a useful strategy for determing the relative significance of terms used in the user reviews by weighing the frequency of their appearances within a review against their relative rarity across all reviews in consideration.

We will also factor in `# thumbsUpCount` in our modeling, so that our models can assign more significance to reviews that received more thumbs ups from other users, as we can reasonably assume that those reviews resonated with others.

### Targets
The `# score` column will be used as our Target column by combining the low ratings (1 and 2) as a Negative class and combining the positive reviews (4 and 5) as a Positive class, and turning these classes into binaries: 0 for Negative, 1 for Positive.

We will be disregarding the ratings of 3 as we want a clear understanding of what makes a review strictly negative vs. strictly positive, and a middle of the road review of 3, as insightful as its content may be, would hinder our ability to understand the division of these sentiments.

### Class Distribution
The imbalance in the distribution of our Negative and Positive classes is not too heavy, with a  skew towards Negative.
- Negative: 61%
- Positive: 39%

We can hopefully minimize the impact of this imbalance through the Feature Engineering outlined above, and by tuning the hyperparameters of our models.

### Success Metrics
In order to evaluate the performance of our models, we will be using the F1 Score, which is the harmonic mean between the rates of **False Negatives** and **False Positives** generated by our models. 

It will additionally be useful in this case since False Negatives are not necessarily more detrimental to our purposes than False Positives, since as we argued above, both Positive and Negative reviews are of value to us.

Due to the slight class imbalance described above, this score would be more useful than a more general Accuracy score.

## Data Preparation

# Modeling

# Conclusion

## Evaluation

## Limitations

## Recommendations

## Next Steps

# Appendix
## Navigation
