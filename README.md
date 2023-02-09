# 🤖 tinyGPT
The best way to understand ChatGPT is to... code it yourself! This project includes an implementation of a General Pre-trained Transformer, according to the iconic [Attention is All You Need paper](https://arxiv.org/abs/1706.03762). However, there are two main discrepancies:

1) This is a decoder only. The encoder part is not implemented and is still a to-do,

2) This decoder has a pre-norm formulation. This means data normalization is performed before the attention block, not after.

My goal here was to better understand this type of NLP model. This one allows for a character-by-character generation of texts from an `input.txt` file. I tried this out on a dataset of [40k lines of Shakespeare](https://huggingface.co/datasets/tiny_shakespeare) which resulted in english-like Shakespearian plays. Seeing how the model generated gibberish at first, I consider this a success.

This project also includes a Bigram language model, however it was created to see the progression from a simple to a more sophisticated model.

## Key learnings

### 1. Self-attention
The `transformers.ipynb` notebook demonstrates various kinds of self-attention mechanisms, from simple to more advanced ones. It is used for predicting the next input. In order to do that, we need to levarage the character context, while simultaneously preventing the using of future inputs - these are the ones we want to predict, after all. The simplest way to create a self-attention mechanism is to average out the characters that preceded the target (cell 3). In order to make it applicable to a batch matrix, we can use a lower triangular matrix divided by the sum of it's rows (cell 4). We can further apply the softmax function and levarage it's averaging capabilities (cell 5). However, if we want to fully implement the self-attention mechanism from the paper, we need to use the concept of key, value and query (cell 6).

### 2. Key, value, query
The goal of these is to implement the attention mechanism. There are two possible explanations. One is that the query relates to what the current token is looking for, the keys are what other tokens are offering, and the value is what the current token is offering to other tokens. In this way a vowel may look for a consonant, and so thanks to it's query and other letters' keys it can calculate the attention it should give to other tokens. Another way to explain keys, queries and values is by referencing databases. A typical database consists of keys and values, which we can access via queries. We can think of querying as multiplying queries times keys. This operation get't us an attention map, with relevant values having the highest attention. If we multply this by the values, we get the result of the database query.

### 3. The importance of normalization
In the attention head (line 23 of `tinyGPT.py`), while multiplying the keys and queries, a normalization is performed. The result of the multiplication is divided by the square root of the target's channels. This is done on purpose, since the next step is to apply the softmax function. If normalization wasn't performed, the softmax function would output an almost one-hot encoded vector because of large differences between the highest and lowest values.

### 4. The phases of attention blocks
In the attention block (line 85 of `tinyGPT.py`), we can interpret the self-attention phase as a communication phase between the different tokens. The feedforward layer acts as a way to compute and retrieve important information from the previous phase for the tokens.

## Acknowledgements
I want to thank [karpathy](https://github.com/karpathy) for his excellent [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY) video on this topic. This project is based on it.
