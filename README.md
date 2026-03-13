# CS 221M: Mechanistic Interpretability

Stanford University, Spring 2026

Course website: [https://cs221m.github.io/](https://cs221m.github.io/)

## Syllabus

| Week        | Date          | Lecture                                    | Notebook                                                                                                               |
| ----------- | ------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| **Week 1**  | Mon. March 30 | Introduction                               |                                           |
|             | Wed. April 1  | Review of language models                  | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cs221m/cs221m-course/blob/main/02_review_of_lms.ipynb) |
| **Week 2**  | Mon. April 6  | Behavioral analysis and input attribution  |                                           |
|             | Wed. April 8  | Probes for decoding activations            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cs221m/cs221m-course/blob/main/04_probes.ipynb) |
| **Week 3**  | Mon. April 13 | Interventions for steering activations     | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cs221m/cs221m-course/blob/main/05_interventions.ipynb) |
|             | Wed. April 15 | Causal mediation analysis                  | —                                         |
| **Week 4**  | Mon. April 20 | Theory of causal abstraction I             | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cs221m/cs221m-course/blob/main/07_causal_abstraction_i.ipynb) |
|             | Wed. April 22 | Designing counterfactuals                  | —                                         |
| **Week 5**  | Mon. April 27 | Automated causal interpretability          | —                                         |
|             | Wed. April 29 | Theory of causal abstraction II            | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cs221m/cs221m-course/blob/main/10_causal_abstraction_ii.ipynb) |
| **Week 6**  | Mon. May 4    | —                                          | —                                         |
|             | Wed. May 6    | —                                          | —                                         |
| **Week 7**  | Mon. May 11   | —                                          | —                                         |
|             | Wed. May 13   | —                                          | —                                                                                                                      |
| **Week 8**  | Mon. May 18   | —                                          | —                                                                                                                      |
|             | Wed. May 20   | —                                          | —                                                                                                                      |
| **Week 9**  | Mon. May 25   | —                                          | —                                                                                                                      |
|             | Wed. May 27   | —                                          | —                                                                                                                      |
| **Week 10** | Mon. June 1   | Project presentations                      | —                                                                                                                      |
|             | Wed. June 3   | Project presentations                      | —                                                                                                                      |

## Learning goals

#### 1. Introduction
 - Overview of course 

#### 2. Review of language models
 - Reviewing fundamentals of language models with an eye towards understaning their components
 - Autoregressive modeling - activations can only be affected by previous tokens
 - Residual stream, attention (moving information between tokens), and MLP (processing/recalling information)

#### 3. Behavioral analysis and input attribution
 - Minimal pairs for understanding model behavior
 - Input attribution - integrated gradients

#### 4. Probes for decoding activations
 - Survey probing methods: logit lens (pre-initialized probe), PCA (unsupervised), and supervised linear probes
 - Introduction to constructing interventions from probes (steering vectors)

#### 5. Interventions for steering activations
 - Survey different types of interventions an their respective goals
 - Iterative nullspace projection for erasing information
 - Interchange interventions for localizing representations
 - Representation fine-tuning for controlling model behavior

#### 6. Causal mediation analysis
 - Interchange interventions on minimal pairs for tracing information flow
 - Intervening on residual stream, individual attention heads
 - Tracing information flow in factual recall, reference resolution

#### 7. Theory of causal abstraction I
 - Theory of constructive abstraction
 - Example constructive abstraction in hierarchical equality task

#### 8. Designing counterfactuals
 - Guidelines for designing counterfactuals to target specific causal variables
 - Pointer-value mechanism in MCQA,
 - Multi-source interventions and entity vs. relation in RAVEL

#### 9. Automated causal interpretability
 - Introduction to distributed alignment search
 - Disentangling entity and relation in RAVEL

#### 10. Theory of causal abstraction II
 - Theory of abstraction under translation
 - Walkthrough example of DAS on hierarchical equality