# CHFEN: Conditional Hybrid Fusion Emotion Net for Short Video Affective Computing


## Project Overview
This repository presents the **CHFEN (Conditional Hybrid Fusion Emotion Net)**, a deep learning model designed for analyzing multi-modal affect in **User-Generated Content (UGC)** short videos.

**CHFEN** represents my early exploration into multimodal learning. Its core value lies in two major innovations:

1.  **Theoretically Grounded Dataset Construction:** Incorporating **Plutchik's Wheel of Emotions** into the annotation framework to create a short video affective dataset with greater **fine-grained explainability**.
2.  **Three-Layer Hybrid Fusion Model:** Designing a fusion network based on the **Encoder-Only** architecture. It leverages a **Conditional Query Mechanism** to efficiently process and integrate multi-modal information (visual, acoustic, and text).

**Project Context:** This work was my first deep exploration into multimodal learning and was submitted (but not accepted) to CVPR [2025]. Despite the model's performance being limited by the era's context and the early dataset quality, this project comprehensively documents the full pipeline—from data crawling and theoretical framework establishment to model training. It serves as an essential **milestone** and a valuable **summary of engineering experience** in the field of multimodal deep learning.


## Core Work I: Dataset Construction and Theoretical Advancement
This project aimed to address the limitations of existing affective datasets by constructing a UGC short video emotion dataset that is closer to the real world and more theoretically sound.

### Bridging the Gap in Real-World Affective Datasets
Existing multimodal emotion datasets are predominantly based on movies and TV shows, lacking data specific to UGC scenarios like news short videos. This presents two challenges:
- **Challenge:** Models trained on conventional datasets often suffer from poor **generalization ability** when applied to high-noise, real-world UGC content.
- **Contribution:** We constructed a **UGC short video dataset** via web crawling, designed to capture more authentic and time-sensitive affective information.

### Theory-Based Annotation Framework: Plutchik's Wheel
The coarse, single-label annotations in most datasets often ignore semantic differences or conflicts across modalities (visual, acoustic, text). To resolve this, we established a new theoretical annotation system:
- **Theoretical Foundation:** We introduced **Plutchik's Wheel of Emotions** from psychology to build a universal and extensible annotation system.
- **Explainability:** This framework better accommodates **multi-modal label conflict scenarios**, providing more **fine-grained, multi-dimensional** explanatory labels than a single-tag approach.
- **Universality and Extensibility:** It provides robust psychological theory to support annotation, allowing for flexible **dimensionality reduction** or **fine-grained extension** based on specific task requirements.

### Multimodal Data Processing and Feature Extraction Pipeline
To support model input and the theoretical annotation framework, we built the following data processing workflow:
- **Crawling and Data Cleaning:** Initial processing and filtering of crawled results based on video metadata.
- **Data Augmentation and Feature Generation:** Feature extraction and pre-processing for video and audio modalities.
    - **Audio Processing:** Separation of background music from human voice.
    - **Text Extraction:** Using recognition algorithms to extract **hard subtitles** (all on-screen text) into `.srt` files.
    - **Visual Tracking:** Extraction and tracking of unique individuals in the video.
- **Multi-Round Manual Annotation:** Multi-dimensional, fine-grained affective annotation was outsourced to an external studio, following the established theoretical framework and strict quality control procedures.


## Core Work II: CHFEN Model Architecture and Multimodal Fusion Strategy
**CHFEN** adopted the popular **Encoder-Only** architecture as its foundation and designed a clear three-layer hybrid fusion structure for effective integration of multimodal information.

### The Three-Layer Hybrid Fusion Architecture
**CHFEN** employs a **hybrid fusion strategy**, integrating multi-modal data at both the feature and decision levels:
- **Encoding Layer:** Modality-specific feature extraction. Utilizes Transformer-based Encoder-Only pre-trained models for independent embedding of each modality.
- **Feature Fusion Layer:** **Low-level representation fusion**. Fuses representations of more tightly coupled modalities (e.g., Text-Visual, Visual-Acoustic).
- **Decision Fusion Layer:** **Task decision integration**. Performs the final integration of all fused features at a higher level to output the final task result (emotion classification).

### Model Construction Rationale
- **Core Mechanism:** Leveraging the superiority of **Attention** in **Cross-Attention** for its natural fit in multi-modal cross-retrieval and information fusion.
- **Modality Encoding:** Utilizing single-modality pre-trained models as a base to ensure high quality in initial representations.
- **Motivation for Hybrid Fusion:** Fusion occurs not only at the low-level feature layer but also through comprehensive consideration at the final decision layer, ensuring more complete multi-modal information integration and enhancing the model's **robustness**.

### Conditional Query-Based Fusion Mechanism
To address the pain points of **modality imbalance** and **inefficient cross-information learning** in simple fusion, **CHFEN** introduced a targeted guidance mechanism:
- **Pain Point:** Simple fusion often fails to learn effective cross-information and is prone to modality weight imbalance (e.g., visual information dominating the weights).
- **Mechanism:** We introduced a global representation called the **Conditional Query** to purposefully guide the low-level modality fusion in early stages.
- **Implementation:** This global representation was specifically chosen as the **news title embedding** (from the multimodal embedding model's encoder). This provides **stable, preferential a-priori guidance** for subsequent temporal video information fusion.


## Lessons Learned & Reflection
This project was my first practical attempt at deep learning research and submitting an AI paper. The challenges exposed are crucial for understanding limitations in real-world application.

**Core Lesson: Data Quality is the Upper Bound for any Deep Learning Task.**

The model's sub-optimal final performance is primarily attributed to limitations in **Data Quality** and **Methodology**:

### Data Quality Bottlenecks and Engineering Challenges
- **High UGC Data Noise:** The inherent noise in short video data (quality, editing, music) interfered significantly with feature extraction.
- **Lack of Data Pre-Analysis:** Failure to adequately pre-analyze the dataset's characteristics before complex processing and annotation led to incomplete cleaning.
- **Insufficient Annotation Quality Control:** Early annotation processes were not rigorous enough, resulting in severe issues with label **credibility** and **consistency**, which undermined the training of a complex model.

### Methodological Limitations
Post-2022, a simple Encoder-Only architecture trained from scratch or via simple transfer learning can no longer compete with the strong **generalization capabilities** of the rising **Multimodal LLMs** and **Embedding Models**.

*Summary: A skilled cook cannot make a meal without rice. A poor dataset (in terms of both raw data and label quality) is an insurmountable barrier for any deep learning project.*


## Project Structure
This structure reflects my early learning curve in PyTorch componentization and Python package control.

- `configs`: Configuration files. **Issue:** Dispersed and redundant configuration. **Improvement:** Adopt a centralized, modular configuration management approach.
- `data_processing`: Scripts for crawling result processing, data cleaning, and simple analysis.
- `dataset`: Data loading and temporal information integration. **Issue:** Encoding performed during the loading stage, leading to inefficiency. **Improvement:** Pre-encode data into **Tensor** format and cache it after initial analysis/processing. (This requires better data class control and is necessary for proper ablation studies).
- `embedding`: Feature extraction based on pre-trained models.
- `model`: PyTorch model definition, built as components and layers.
- `utils`: General utility functions not natively implemented in PyTorch.
- `tests`: Simple tests based on IPynb. **Improvement:** Adopt **pytest** for standard unit and integration testing.
- `scripts`: Model execution scripts.


## How to Run

1.  Build the example configuration file, e.g., `baseline_model.yaml`, in the `config_experiments` folder.
2.  Set `main.py` to use the corresponding configuration file.
3.  Run `main.py` (you can use `nohup` for background execution).


## Planned Publication
**TODO:** I plan to release the original paper related to this project on Arxiv after refining my scientific writing approach.


## More Projects and Current Work
This project is a record of growth, but its standards for methodology and best practices no longer meet my current requirements. If you are interested in more modern, efficient deep learning and AI engineering practices, please refer to my other projects.

### Related/Improved Work
In my view, encoder-only models are limited to simpler tasks. Post-2022, unless there is significant domain-specific accumulation, the best approach is to leverage new LLM-related technologies.
- [MEM-V-Agent](https://github.com/yuliu625/MEM-V-Agent): The planned follow-up work to this project, aiming to build a multimodal analysis agent to accomplish the task. Due to various practical reasons, this work remains incomplete. All my research materials and code are in this repository, should you wish to explore them.

### My Current Deep Learning Projects
- [Flash-Boilerplate](https://github.com/yuliu625/Yu-Flash-Boilerplate): My latest deep learning project repository template, featuring greater standardization and modularization.
- [Deep-Learning-Toolkit](https://github.com/yuliu625/Yu-Deep-Learning-Toolkit): A repository of common deep learning tools I have built.

### More Utility Repositories
- [Data-Science-Toolkit](https://github.com/yuliu625/Yu-Data-Science-Toolkit): My toolkit for data science tasks.
- [Agent-Development-Toolkit](https://github.com/yuliu625/Yu-Agent-Development-Toolkit): A toolkit focused on LLM and Agent construction.

### Existing Research Work
- [Simulate-the-Prisoners-Dilemma-with-Agents](https://github.com/yuliu625/Simulate-the-Prisoners-Dilemma-with-Agents): My early attempt using `autogen` to study LLM agent behavior in simple game theory scenarios like the Prisoner's Dilemma.
- [World-of-Six](https://github.com/yuliu625/World-of-Six): Research on agent decision-making behavior in environments with network effects. (Paper accepted by SWAIB[2025])

### Ongoing Projects (Future Open-Source)
- Research on the expected behavior of LLM-based Agents in environments featuring network effects.
- A document intelligence project building a Multi-agent System for financial report analysis.

Feel free to check out my other work and connect with me for discussion!

