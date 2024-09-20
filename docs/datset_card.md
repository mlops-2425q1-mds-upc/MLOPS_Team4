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

### Dataset Description

<!-- Provide a longer summary of what this dataset is. -->


- **Shared by:** Alec Go, Richa Bhayani, Lei Huang
- **Language(s) (NLP):** Informal English (it contains misspellings and slang)
- **License:** Open license

### Dataset Sources 
Dataset comes from the paper "Twitter Sentiment Classification using Distant Supervision"
and training and test datasets can be downloaded from https://twittersentiment.appspot.com/.

## Uses

### Direct Use
This dataset is suitable for sentiment analysis and natural language processing

### Out-of-Scope Use
The dataset is focused on English tweets so conclusions may not be extrapolable for other languages.
Moreover, tweets are obtained in some domains and on a given date, so this dataset
can not be used for specific domains or to make a temporal study.

## Dataset Structure

<!-- This section provides a description of the dataset fields, and additional information about the dataset structure such as criteria used to create the splits, relationships between data points, etc. -->

[More Information Needed]

## Dataset Creation

### Curation Rationale
Sentimental analysis studies are important for 
customers as they can review products before
buying and for marketers, which can be informed about
the general opinion of their products. Despite that,
nowadays sentimental analysis is mostly done on reviews,
without considering other sources like Twitter. In this platform, users can
reveal their opinions in a more informal way than
in a review, moreover, it is easy to interact with other
users to complement their views.

<!-- Motivation for the creation of this dataset. -->

[More Information Needed]

### Source Data

<!-- This section describes the source data (e.g. news text and headlines, social media posts, translated sentences, ...). -->

#### Data Collection and Processing

<!-- This section describes the data collection and processing process such as data selection criteria, filtering and normalization methods, tools and libraries used, etc. -->

[More Information Needed]

#### Who are the source data producers?

<!-- This section describes the people or systems who originally created the data. It should also include self-reported demographic or identity information for the source data creators if this information is available. -->

[More Information Needed]

### Annotations [optional]

<!-- If the dataset contains annotations which are not part of the initial data collection, use this section to describe them. -->

#### Annotation process

<!-- This section describes the annotation process such as annotation tools used in the process, the amount of data annotated, annotation guidelines provided to the annotators, interannotator statistics, annotation validation, etc. -->

[More Information Needed]

#### Who are the annotators?

<!-- This section describes the people or systems who created the annotations. -->

[More Information Needed]

#### Personal and Sensitive Information

<!-- State whether the dataset contains data that might be considered personal, sensitive, or private (e.g., data that reveals addresses, uniquely identifiable names or aliases, racial or ethnic origins, sexual orientations, religious beliefs, political opinions, financial or health data, etc.). If efforts were made to anonymize the data, describe the anonymization process. -->

[More Information Needed]

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

[More Information Needed]

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users should be made aware of the risks, biases and limitations of the dataset. More information needed for further recommendations.

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



## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the dataset or dataset card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Dataset Card Authors [optional]

[More Information Needed]

## Dataset Card Contact

[More Information Needed]