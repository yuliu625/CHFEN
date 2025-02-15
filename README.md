# CHFEN: Conditional Hybrid Fusion Emotion Net for Short Video Affective Computing

## Overview
Here is an article I once tried to submit to CVPR[2025]. At that time, I was interested in multi-modality. So, I designed a neural network that could handle data from multiple modalities.

The purpose of this network is to analyze the emotions shown in a video. 
I built some crawlers to crawl  a lot of videos from short video platforms. Then I build a data pipeline to process, analyze and label the dataset. 
The modules in the network are based on some popular encoder-only models that I could find at that time. The network goes through various analysis, processing, encoding and fusion to finally output the analysis result.

However, due to the poor quality of the dataset itself(characteristic of UGC), as well as the poor quality of the annotation(I actually did a lot of experiments later to prove this.), it is very difficult to train the network on such a bad dataset. At the same time, encoder-only models were outdated at that point. In particular, without the use of transfer learning approaches, it will always be difficult to regulate how the data and the model scale and relate to each other. (Then the use of multimodal LLMs as well as multimodal embedding models will be a much better choice. So after this submission, I changed my research direction to some extent.) Because of these reasons, the model didn't work out very well in the end.

In retrospect, the project looks really terrible. I have since been able to build much higher quality projects with better tools, as can be seen in the projects in my other repositories. Overall, though, this contribution was my first complete project on deep learning network.

## Composition

### configs

### data_processing

### dataset

### embedding

### model

### scripts

### utils


## How it works


## Summary

