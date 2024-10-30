# Manuscript

## Introduction
Key problems to be asked for every project:
1.  What challenges & problems do we solve? Why is it important?
2.  How does the proposed method work to solve these challenges? What are the differences from previous works?
3. What are the insights or contributions of the proposed work to the research community?
### Problem & Key Challenges
Current approaches still have problems with:
- Current work in multimodal affective computing focuses on the emotions represented by multiple modalities of video together, rather than the emotional respose of the video viewer.
- The datasets are mostly from artistic renditions such as movies and TV shows, not from the real world.
- The dataset is labeled arbitrarily, mostly for the convenience of the deep learning task, and lacks a theoretical foundation. 
  Labeling categories are few and connections between labels are ignored.
- The part of modal fusion is not well handled and does not specifically consider the semantic information of multiple modalities. 
  Modal fusion is done using only feature fusion or decision fusion without considering that different fusion strategies should be used for different modalities with a high degree of similarity and a low degree of similarity.
### Contributions
- **Dataset:** DNSV dataset
  - **Pipeline:** 
    Selected from real news data of short video platforms, a series of data cleaning and construction of relevant data were carried out.
  - **Plutchik's Wheel of Emotions:** 
    The choice of data labels is based on the psychological theory of Plutchik's Wheel of Emotions, which is more theoretically grounded and realistic as well as interpretable. 
    The existing results can be easily transformed and migrated. 
    And there is still room for future research toward continuous emotion modeling and emotion intensity.
  - **Feedback-based annotation:** 
    The result of data annotation is the viewer's emotional response, which is more realistic and reliable, and has more practical value.
- **Network:** CHFEN Conditional Hybrid Fusion Emotion Net
  - **Enhanced modality interactions:** 
    Using news title with a significant level of profile as a query to enhance feature extraction.
  - **Balanced the inherent differences between image and text modality:** 
    Using BLIP captioner to generate text data makes the text modality more reliable. The whole net is more balanced.
  - **Character and scene equalization:** 
    With the density of characters throughout the video, it is more reasonable to integrate these two image modalities.

## Motivation
**A reasonable motivation is extremely important.**
Will focus on introducing new methods to solve the key challenges. Current ideas might include:
### Motivation:
1. Existing ways of affective computing of a video only analyze the emotion of the video itself in the interpreted scene, which itself can be obtained through the theme of the script. 
   Artistic interpretation of a single scene, the number of characters with a certain relationship and emotional performance is obvious. 
   While the real scene is complex and changeable, the character relationship is uncertain, and the interpretation scene is very different from the reality. 
   Short news video will not be limited to a single scene, will specifically show the location of the news and the events. 
   At the same time, the news title is more generalized and professional, which is more representative and meaningful than the general video title.
2. Existing emotion annotations are simply a small number of labels, while not reasonable for categorizing emotion. 
   Fewer labels would be easier to obtain higher accuracy for the classification task, but the practical value would also be reduced. 
   At the same time, different classification methods are not compatible with each other. 
   For example, the models of 3-categorization and 5-categorization cannot be simply transferred or interpreted with each other. 
   However, in fact, there are mutual correlations between different emotions, which should not be multiple one-hot vectors. 
   For the selection of emotion labels, psychological theories should be considered at the time of labeling, so that the evaluation of the results will be more reasonable. 
   We need a dataset which has more types of emotion labels and at the same time these labels have the interpretability of psychological theories.
3. Often times, the purpose of our affective computing is to get feedback and evaluation. 
   The emotion of a scene and the emotion that the actual scene ultimately conveys to people are two tasks that should not be confused. 
   In practical applications, especially news, it is necessary that we can predict the feelings that this news brings to the people before releasing the news, so as to evaluate the spread and influence of this news. 
   This is a very important guide and practical value for the modern media industry.
### References:
- Koala: Key frame-conditioned long video-LLM
- How you feelin’? Learning Emotions and Mental States in Movie Scenes
- M2FNet: Multi-modal Fusion Network for Emotion Recognition in Conversation
### Prilimery:


## Methodology


## Experiments
### Datasets
- DNSV
- MovieGraphs
### Testing Benchmarks
- EmoTx
- M2FNet
### Comparison

### Ablation Studies
- Unimodality
  - **Audio:** What if without audio modality?
- Multimodality
  - **Modality interactions:**
    - **Cross attention:** 
      Whether the introduction of cross attention can improve the performance of unimodality?
    - **Blip image captioner:** 
      Whether the captioner can improve the performance of text modality?
- Module
  - **Conditioned image encoder:**
    Effectiveness of fusion of person and scence embedding by person density.
  - **Conditioned text encoder:**
    Effectiveness of captioner.
### Model Analysis
visual analysis

quantitative comparison

### Applications

