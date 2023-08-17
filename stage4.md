---
title: Text Classification and Text Generation using Recurrent Neural Network
layout: default
nav_order: 4
---
# Text Classification and Text Generation using Recurrent Neural Network
{: .no_toc}
Cleaned the text data, removed stop words and punctuations and also normalized the words. Extracted the vocabulary set from the provided data. Trained a RNN model to classify the documents into either positive or negative classes with the training set.

Trained a RNN model, which can generate the whole piece of a sentence based on a given input of the beginning three words.

## Table of contents
{: .no_toc .text-delta}

1. TOC
{:toc}

## Dataset
[link provide later](https://www.example.com)

This dataset contains movie reviews along with their associated binary
sentiment polarity labels. It is intended to serve as a benchmark for
sentiment classification. This document outlines how the dataset was
gathered, and how to use the files provided. 

### Text Classification
The core dataset contains 50,000 reviews split evenly into 25k train
and 25k test sets. The overall distribution of labels is balanced (25k
pos and 25k neg).

In the entire collection, no more than 30 reviews are allowed for any
given movie because reviews for the same movie tend to have correlated
ratings. Further, the train and test sets contain a disjoint set of
movies, so no significant performance is obtained by memorizing
movie-unique terms and their associated with observed labels.  In the
labeled train/test sets, a negative review has a score <= 4 out of 10,
and a positive review has a score >= 7 out of 10. Thus reviews with
more neutral ratings are not included in the train/test sets.

## Abstract
Familiarized with the recurrent neural network (RNN) model, and used the model to classify and generate text data. First, we cleaned the dataset given and then tokenized before passing the data through BERT embedding. Next, we use the Pytorch library to train RNN models by passing in the BERT embedding into it. To test for classification, we could compare the expected output. However, for the generation component, we tested it with our own input