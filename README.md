# 🤖 tinyGPT

The best way to understand ChatGPT is to... code it yourself! This project includes an implementation of a General Pre-trained Transformer, according to the iconic [Attention is All You Need paper](https://arxiv.org/abs/1706.03762). However, there are two main discrepancies:

1) This is a decoder only. The encoder part is not implemented and is still a to-do,

2) This decoder has a pre-norm formulation. More on that in the section below.

My goal here was to better understand this type of NLP model. This one allows for a character-by-character generation of texts from an `input.txt` file. I tried this out on a dataset of [40k lines of Shakespeare](https://huggingface.co/datasets/tiny_shakespeare) which resulted in english-like Shakespearian plays. Seeing how the model generated gibberish at first, I consider this a success.

## Key learnings

to-do

## Acknowledgements

I want to thank [karpathy](https://github.com/karpathy) for his excellent [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY) video on this topic. This project is based on it.
