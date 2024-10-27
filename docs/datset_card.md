---
license:
- no known license
task_categories:
- text-classification
language:
- en
tags:
- sentimental analysis
- twitter
pretty_name: sentiment140
size_categories:
- 1M<n<10M
---
# Dataset Card for sentiment140


<!-- Provide a quick summary of the dataset. -->


This dataset card aims to be a base template for new datasets. It has been generated using [this raw template](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/templates/datasetcard_template.md?plain=1).


## Dataset Details
This is the sentiment140 dataset. It contains 1,600,000 tweets extracted using the Twitter API.
### Dataset Description


<!-- Provide a longer summary of what this dataset is. -->
Sentiment140 dataset provides 1,600,000 tweets extracted using the Twitter API. Each tweet has been labeled with a polarity score (0 - negative, 2 - neutral, 4 - positive) and they can be used to detect the sentiment of the tweet.
The dataset includes the following 6 fields:
target
ids
date
flag
user
text


- **Shared by:** Alec Go, Richa Bhayani, Lei Huang
- **Language(s) (NLP):** Informal English (it contains misspellings and slang)
- **License:** Open license


### Dataset Sources
The dataset comes from the paper "Twitter Sentiment Classification using Distant Supervision". It can be downloaded from https://twittersentiment.appspot.com/.


## Uses


### Direct Use
This dataset is suitable for sentiment analysis and natural language processing


### Out-of-Scope Use
The dataset is focused on English tweets so conclusions may not be extrapolable for other languages. Moreover, tweets are obtained in some domains and on a given date, so this dataset can not be used for specific domains or to make a temporal study.


## Dataset Structure


<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->
Here's a breakdown of the dataset structure based on its key features:
Size: 1.6 million tweets, annotated as positive, negative, or neutral.
Fields:
target: The sentiment polarity of the tweet (0 = negative, 2 = neutral, 4 = positive).
ids: Unique identifier for the tweet (e.g., 2087).
date: The timestamp when the tweet was created (e.g., Sat May 16 23:58:44 UTC 2009).
flag: A query field that shows if a specific query was used to retrieve the tweet (if no query, it will show NO_QUERY).
user: Username of the person who posted the tweet (e.g.,
scotthamilton).
text: The tweet itself (e.g., “is upset that he can't update his Facebook by texting it... ”).
Each tweet is independent, with no direct relationship to other tweets. However, users can group tweets by user or query to perform specific analyses, such as user-level sentiment analysis or topic-specific sentiment classification




## Dataset Creation


### Curation Rationale
Sentimental analysis studies are important for customers as they can review products before buying and for marketers, which can be informed about the general opinion of their products. Despite that, nowadays sentimental analysis is mostly done on reviews, without considering other sources like Twitter. In this platform, users can reveal their opinions in a more informal way than in a review, moreover, it is easy to interact with other users to complement their views.


<!-- Motivation for the creation of this dataset. -->


[More Information Needed]


#### Data Collection and Processing


<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. →
Tweets are from April 6, 2009 to June 25, 2009.
Any tweet containing both positive and negative emoticons is removed. This may happen if a tweet contains two subjects.


Emoticons are stripped off.


Retweets are removed. Retweeting is the process of copying another user’s tweet and posting it to another account. This usually happens if a user likes another user’s tweet. Retweets are commonly abbreviated with “RT.” Any tweet with RT is removed from the training data to avoid giving a particular tweet extra weight in the training data.
4. Tweets with “:P” are removed. At the time of data retrieval, the Twitter API has an issue in which tweets with “:P” are returned for the query “:(”. These tweets are removed because “:P” usually does not imply a negative sentiment.
5. Repeated tweets are removed. Occasionally, the Twitter API returns duplicate tweets. The scraper compares a tweet to the last 100 tweets. If it matches any, then it discards the tweet. Similar to retweets, duplicates are removed to avoid putting extra weight on any particular tweet.


#### Who are the source data producers?


<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->


Source data producers are Twitter platform users who created and posted the tweet.


#### Annotation process


<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->


#### Who are the annotators?
Annotations of the target are done using a classifier trained on emoticon data against a test set of tweets


#### Personal and Sensitive Information
This dataset contains only tweets of public usernames of Twitter, so users allow anyone to retrieve them. They do not contain any private or sensitive data.


<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->


## Bias, Risks, and Limitations
<!-- This section is meant to convey both technical and sociotechnical limitations. -->
Tweets were collected only between April and June, which limits the generalizability of the results to that specific time period
Dataset is focused only on English tweets, which may exclude or underrepresent non-English speaking Twitter users.


### Recommendations
<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->


Users should be made aware of the risks, biases and limitations of the dataset. For example, it only considers tweets from some public accounts and related to some topics.


## Citation [optional]
<!-- If there is a paper or blog post introducing the dataset, the APA and Bibtex information for that should go in this section. -->


**BibTeX:**


@article{go2009twitter,
  title={Twitter sentiment classification using distant supervision},
  author={Go, Alec and Bhayani, Richa and Huang, Lei},
  journal={CS224N project report, Stanford},
  volume={1},
  number={12},
  pages={2009},
  year={2009}
}
