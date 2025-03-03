# CHFEN: Conditional Hybrid Fusion Emotion Net for Short Video Affective Computing

## Overview
Here is an article I once tried to submit to CVPR[2025]. At that time, I was interested in multi-modality. So, I designed a neural network that could handle data from multiple modalities.

The purpose of this network is to analyze the emotions shown in a video. 
I built some crawlers to crawl  a lot of videos from short video platforms. Then I build a data pipeline to process, analyze and label the dataset. 
The modules in the network are based on some popular encoder-only models that I could find at that time. The network goes through various analysis, processing, encoding and fusion to finally output the analysis result.

However, due to the poor quality of the dataset itself(characteristic of UGC), as well as the poor quality of the annotation (I actually did a lot of experiments later to prove this.), it is very difficult to train the network on such a bad dataset. At the same time, encoder-only models were outdated at that point. In particular, without the use of transfer learning approaches, it will always be difficult to regulate how the data and the model scale and relate to each other. (Then the use of multimodal LLMs as well as multimodal embedding models will be a much better choice. So after this submission, I changed my research direction to some extent.) Because of these reasons, the model didn't work out very well in the end.

In retrospect, the project looks really terrible. I have since been able to build much higher quality projects with better tools, as can be seen in the projects in my other repositories. Overall, though, this contribution was my first complete project on deep learning network.

## Composition
This was my practice at that time for designing the project structure. As the project get bigger, designing packages is necessary for reusability and better management. However, this is not a good design. Later, I realized a better way to manage torch-related projects.

### configs
This is related to the project configuration. At that time, I had thought that I could put all configuration files together. Unified format, unified management. However, managing setting requires sufficient understanding of the project itself. Too decentralized and too centralized configurations tend to complicate configuration. It can be seen that I made improvements in `config_experiments`, but I still don't understand that there are relationships and inheritance between configuration files. As a result, it was also tedious to do experimental configurations.

### data_processing
This package is part of the dataset pipeline I constructed. There is actually more code, but this is all that is organized. Dataset processing has a lot of reusable operations. So I have since constructed som repositories to organize and summarize these operations.

### dataset
Here is the part of the dataset for this model. In most case, building a Pytorch dataloader is not a difficult problem. However, since this is a multi-modal dataset, it can be a pain to go through and coordinate the loading and processing of different types of data. Especially for the ablation study and comparison experiments that will be done later, and for loading other datasets for generalization, the module has to be well thought and the code quality has to be good enough at the beginning (Although it seems to me that the code quality is not good enough, even though I did put a lot of effort into building this module at that time.) I provide 3 ways to build the dataset,  though only one way was actually used in the model.

### embedding
This is the fundamental part of the model, encoding the raw modal data into embeddings. The encoders I provide are basically encapsulations of encoder-only models in `transformers`, so they can be easily loaded and used locally. The difficulty in implementing this module is that there are many temporal issues with the individual modalities. Some modalities are global, some are in strict correspondence with each other, and some are continuous and difficult to slice. I construct a `total_encoder` to represent my implementation.

To adapt to different datasets, I build this module as a tool for the dataset package. In practice, however, it is possible to store the embeddings directly serialized for convenience and speed. Some paper do this, although multiple slices may be required due to different standards. It would be storage intensive, but it would speed things up a lot.

### model


### scripts
Some shell scripts.

Start the training code with `nohup`. I have since trained with better tools and created automation scripts that encapsulate the shell command operations.

### utils


## How to run


## Summary

