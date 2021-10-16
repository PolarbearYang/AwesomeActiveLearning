# Awesome Active Learning

Active learning (AL) is a machine learning paradigm that take both model and data into consideration. In more detail, AL selects the samples that are the most informative to its models, interacts with external information sources to query the labels of these samples, trains its models over all the existing labeled samples, and repeats this cycle in an iterative manner.
As an easy-to-use framework, the AL strategy has been adopted to several tasks and has demonstrated its power by simplifying data acquisition and reducing annotation cost in various scenarios.
This document presents Awesome Active Learning researches in 3 years.

## CV

### Classification

- Active Generative Adversarial Network for Image Classification (AAAI-2019)
- Bounding Uncertainty for Active Batch Selection (AAAI-2019) **[ML]**
- Self-Paced Active Learning: Query the Right Thing at the Right Time (AAAI-2019) **[ML]**
- Asking the Right Questions to the Right Users: Active Learning with Imperfect Oracles (AAAI-2020)
- ACTIVETHIEF: Model Extraction Using Active Learning and Unannotated Public Data (AAAI-2020)
- Cost-Accuracy Aware Adaptive Labeling for Active Learning (AAAI-2020) **[ML]**
- Online Active Learning of Reject Option Classifiers (AAAI-2020) **[ML]**
- AutoDAL: Distributed Active Learning with Automatic Hyperparameter Selection (AAAI-2020)
  - This is for hyperparameter search
  - Evaluated on Image Classification
  - General framework
- Active Bayesian Assessment of Black-Box Classifiers (AAAI-2021)
- Agreement-Discrepancy-Selection: Active Learning with Progressive Distribution Alignment (AAAI-2021)
- Nearest Neighbor Classifier Embedded Network for Active Learning (AAAI-2021)
  - Classifier head design for active learning
- Improving Model Robustness by Adaptively Correcting Perturbation Levels with Active Queries (AAAI-2021)
- Unsupervised Active Learning via Subspace Learning (AAAI-2021)
- Learning Loss for Active Learning (CVPR-2019)
- Deep Active Learning for Biased Datasets via Fisher Kernel Self-Supervision (CVPR-2020)
- Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation from a Blackbox Model (CVPR-2020)
- State-Relabeling Adversarial Active Learning (CVPR-2020)
- Sequential Graph Convolutional Network for Active Learning (CVPR-2021)
- Task-Aware Variational Adversarial Active Learning (CVPR-2021)
- Transferable Query Selection for Active Domain Adaptation (CVPR-2021)
- VaB-AL: Incorporating Class Imbalance and Difficulty with Variational Bayes for Active Learning (CVPR-2021)
- Variational Adversarial Active Learning (ICCV-2019)
- Semi-Supervised Active Learning with Temporal Output Discrepancy (ICCV-2021)
- Active Learning with Partial Feedback (ICLR-2019)
- Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (ICLR-2020)
- Active Learning for Probabilistic Structured Prediction of Cuts and Matchings (ICML-2019)
 	- Multi-label Classification
- Bayesian Generative Active Deep Learning (ICML-2019)
- Adaptive Region-Based Active Learning (ICML-2020)
- Active Testing Sample-Efficient Model Evaluation (ICML-2021)
- Improved Algorithms for Agnostic Pool-based Active Classification (ICML-2021)
- Message Passing Adaptive Resonance Theory for Online Active Semi-supervised Learning (ICML-2021)
- Deep Active Learning with Adaptive Acquisition (IJCAI-2019)
- Batch Decorrelation for Active Metric Learning (IJCAI-2020)
- Asynchronous Active Learning with Distributed Label Querying (IJCAI-2021)
- BatchBALD: Efficient and Diverse Batch Acquisition for Deep Bayesian Active Learning (NIPS-2019)
- Bayesian Batch Active Learning as Sparse Subset Approximation (NIPS-2019)
- Deep Active Learning with a Neural Architecture Search (NIPS-2019)
- Flattening a Hierarchical Clustering through Active Learning (NIPS-2019)


### Object Detection

- Learning Loss for Active Learning (CVPR-2019)
- Multiple Instance Active Learning for Object Detection (CVPR-2021)
- Active Learning for Deep Detection Neural Networks (ICCV-2019)
- Active Learning for Deep Object Detection via Probabilistic Modeling (ICCV-2021)

### Semantic Segmentation

- Embodied Visual Active Learning for Semantic Segmentation (AAAI-2021)
- State-Relabeling Adversarial Active Learning (CVPR-2020)
- ViewAL: Active Learning With Viewpoint Entropy for Semantic Segmentation (CVPR-2020)
- Revisiting Superpixels for Active Learning in Semantic Segmentation with
Realistic Annotation Costs (CVPR-2021)
- Task-Aware Variational Adversarial Active Learning (CVPR-2021)
- Variational Adversarial Active Learning (ICCV-2019)
- ReDAL: Region-based and Diversity-aware Active Learning for Point Cloud Semantic Segmentation (ICCV-2021)
- Semi-Supervised Active Learning with Temporal Output Discrepancy (ICCV-2021)
- Reinforced active learning for image segmentation (ICLR-2020)
- On Statistical Bias In Active Learning How and When to Fix It (ICLR-2021)
- Uncertainty-aware Active Learning for Optimal Bayesian Classifier (CILR-2021) **[ML]**

### Pose-estimation

- Learning Loss for Active Learning (CVPR-2019)
- Sequential Graph Convolutional Network for Active Learning (CVPR-2021)

### Object Tracking

- Active Learning for Probabilistic Structured Prediction of Cuts and Matchings (ICML-2019)

### Image Generation

- Thinking Outside the Pool Active Training Image Creation for Relative Attributes (CVPR-2019)

### Person Re-Identification

- Deep Reinforcement Active Learning for Human-In-The-Loop Person Re-Identification (ICCV-2019)

### Multi-class Image Classification

- Active Learning of Multi-Class Classification Models from Ordered Class Sets (AAAI-2019) **[ML]**
- Integrating Bayesian and Discriminative Sparse Kernel Machines for Multi-class Active Learning **[ML]**

### Open-Set Classification

- Active Sampling for Open-Set Classification without Initial Annotation (AAAI-2019) **[ML]**

### Action Recognition

- Unsupervised Active Learning via Subspace Learning (AAAI-2021)
- Mindful Active Learning (JICAI-2019)

### Facial Age Estimation

- Unsupervised Active Learning via Subspace Learning (AAAI-2021)

### Audio-video analysis

- Asking the Right Questions to the Right Users: Active Learning with Imperfect Oracles (AAAI-2020)
- Active Contrastive Learning of Audio-Visual Video Representations (ICLR-2021)

### Video Recommendation

- Multi-View Active Learning for Video Recommendation (IJCAI-2019)




## NLP

### Text Classification

- Confidence Weighted Multitask Learning (AAAI-2019) **[ML]**
- Bounding Uncertainty for Active Batch Selection (AAAI-2019) **[ML]**
- Self-Paced Active Learning: Query the Right Thing at the Right Time (AAAI-2019) **[ML]**
- Active Learning with Query Generation for Cost-Effective Text Classification (AAAI-2020) **[ML]**
- ACTIVETHIEF: Model Extraction Using Active Learning and Unannotated Public Data (AAAI-2020)
- Asking the Right Questions to the Right Users: Active Learning with Imperfect Oracles (AAAI-2020)
- Cost-Accuracy Aware Adaptive Labeling for Active Learning (AAAI-2020) **[ML]**
- Online Active Learning of Reject Option Classifiers (AAAI-2020) **[ML]**
- Learning How to Active Learn by Dreaming (ACL-2019)
- Empowering Active Learning to Jointly Optimize System and User Demands (ACL-2020)
- Supporting Land Reuse of Former Open Pit Mining Sites using Text Classification and Active Learning (ACL-2021)
- Uncertainty-aware Active Learning for Optimal Bayesian Classifier (ICLR-2021) **[ML]**
- Active Learning for Probabilistic Structured Prediction of Cuts and Matchings (ICML-2019)
 	- Multi-label Classification
- ActiveHNE: Active Heterogeneous Network Embedding (IJCAI-2019)
- Exemplar Guided Active Learning (NIPS-2019)

### Named Entity Recognition

- MTAAL: Multi-Task Adversarial Active Learning for Medical Named Entity Recognition and Normalization (AAAI-2021)
- Learning How to Active Learn by Dreaming (ACL-2019)
- Active Imitation Learning with Noisy Guidance (ACL-2020)
- Subsequence Based Deep Active Learning for Named Entity Recognition (ACL-2021)

### Keyword/Keyphrase Extraction

- Active Imitation Learning with Noisy Guidance (ACL-2020)

### Sequence/Speech Tagging

- AlpacaTag: An Active Learning-based Crowd Annotation Framework for Sequence Tagging (ACL-2019)
- Active Imitation Learning with Noisy Guidance (ACL-2020)
- Bridge-Based Active Domain Adaptation for Aspect Term Extractio (ACL-2021)

### Entity Resolution

- Learning How to Active Learn by Dreaming (ACL-2019)
- Active Learning for Coreference Resolution using Discrete Annotation (ACL-2020)

### Recommandation System

- Asking the Right Questions to the Right Users: Active Learning with Imperfect Oracles (AAAI-2020)

### Visual Question Answering

- Mind Your Outliers! Investigating the Negative Impact of Outliers on Active Learning for Visual Question Answering (ACL-2021)

### Dialog Policy Learning

- Dialog Policy Learning for Joint Clarification and Active Learning Queries (AAAI-2021)

### Spam Content Detection

- Camouflaged Chinese Spam Content Detection with Semi-supervised Generative Active Learning (ACL-2020)



## Graph

### Node Classification

- Active Learning on Attributed Graphs via Graph Cognizant Logistic Regression and Preemptive Query Generation (ICML-2020)
- Graph Policy Network for Transferable Active Learning on Graphs(NIPS-2020)

### Graph Clustering(Community Detection)

- Active learning in the geometric block model (AAAI-2020) **[ML]**

### Anchor User Prediction

- Deep Active Learning for Anchor User Prediction (IJCAI-2019)




## Others

### Feature Selection

- Active Feature Selection for the Mutual Information Criterion (AAAI-2021) **[ML]**

### Tuple-wise Similarity Learning

- Active Ordinal Querying for Tuple-wise Similarity Learning (AAAI-2020) **[ML]**

### Causal Inference

- Active Learning for Decision-Making from Imbalanced Observational Data (ICML-2019) **[ML]**

### World Model Learning

- Active World Model Learning with Progress Curiosity (ICML-2020)
- Ready Policy One: World Building Through Active Learning (ICML-2020)

### Framework

- An Information-Theoretic Framework for Unifying Active Learning Problems (AAAI-2021) **[ML]**
- Active Learning with Disagreement Graphs (ICML-2019)
- Deeper Connections between Neural Networks and Gaussian Processes Speed-up Active Learning (IJCAI-2019)
- Dual Active Learning for Both Model and Data Selection (JICAI-2021) **[ML]**

### New region

- Active Covering (ICML 2021)
- Cost effective active search (NIPS 2019)
- Beyond the Pareto Efficient Frontier: Constraint Active Search for
Multiobjective Experimental Design (ICML 2021)
- Nonmyopic Multifidelity Active Search (ICML 2021)
- On Deep Unsupervised Active Learning (IJCAI 2020)

### Theory

- Active learning for distributionally robust level-set estimation (ICML 2021)
- The Label Complexity of Active Learning from Observational Data (NIPS-2019)
- Efficient active learning of sparse halfspaces with arbitrary bounded noise (NIPS-2020)
- Finding the Homology of Decision Boundaries with Active Learning (NIPS-2020)
- The Power of Comparisons for Actively Learning Linear Classifiers (NIPS-2020)

## ??

- Active Learning of Continuous-time Bayesian Networks through Interventions (ICML 2021)
- Fast active learning for pure exploration in reinforcement learning (ICML 2021)
- Active Learning within Constrained Environments through Imitation of an Expert Questioner (IJCAI-2019)
- Human-in-the-loop Active Covariance Learning for Improving Prediction in Small
Data Sets (IJCAI-2019)
- Class Prior Estimation in Active Positive and Unlabeled Learning (IJCAI-2020)
- Actively Learning Concepts and Conjunctive Queries under ELr-Ontologies (IJCAI-2021)
- A New Perspective on Pool-Based Active Classification and False-Discovery Control (NIPS-2019)
- Sample Efficient Active Learning of Causal Trees (NIPS-2019)
