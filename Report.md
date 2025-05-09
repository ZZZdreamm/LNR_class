# Session 3. Transformers: Fine-tuning for multi-label classification.

## 1. Data Preprocessing Steps

- **Tokenization**:
  - Used the `AutoTokenizer` from the Hugging Face Transformers library, specific to each model (e.g., BERT, RoBERTa, BETO, RoBERTa-BNE).

- **Label Encoding**:
  - Labels were provided as lists of categories (e.g., ["IDEOLOGICAL-INEQUALITY", "OBJECTIFICATION"]).
  - Used `MultiLabelBinarizer` from scikit-learn to convert these into multi-hot encoded vectors, where each category corresponds to a binary indicator (1 for presence, 0 for absence).


## 2. Model Selection and Architecture Details

Four transformer-based models were selected, two for English and two for Spanish, to leverage their pre-trained knowledge and adaptability to the task.

- **English Models**:
  - **BERT-base-uncased**
  - **RoBERTa-base**

- **Spanish Models**:
  - **BETO (dccuchile/bert-base-spanish-wwm-uncased)**
  - **RoBERTa-BNE (PlanTL-GOB-ES/roberta-base-bne)**

- **Architecture Modifications**:
  - Each model was adapted for multi-label classification by adding a linear classification head on top of the token output
  - The output layer used a sigmoid activation function to produce probabilities for each label independently, suitable for multi-label tasks.
  - For LoRA (Low-Rank Adaptation), we applied low-rank updates to the query and value matrices of the attention layers, significantly reducing the number of trainable parameters while maintaining performance.

The selection balanced model size, language specificity, and computational feasibility, with LoRA enabling efficient fine-tuning.

## 3. Fine-tuning Techniques and Hyperparameter Selection Strategies

Two fine-tuning approaches were employed: standard fine-tuning and LoRA-based fine-tuning,

- **Standard Fine-tuning**:
  - Updated all model parameters, including the pre-trained weights and the classification head.
  - Used the Hugging Face `Trainer` API for streamlined training, evaluation, and checkpointing.

- **LoRA Fine-tuning**:
  - Applied LoRA to reduce the number of trainable parameters, focusing updates on low-rank matrices in the attention layers
  - LoRA configuration:
    - Rank (`r`): 8
    - Alpha (`lora_alpha`): 16
    - Dropout (`lora_dropout`): 0.1
    - Bias: None
  - Benefits: Reduced memory footprint and faster training, ideal for resource-constrained environments.

- **Hyperparameter Selection**:
  - Conducted a grid search over the following ranges:
    - **Batch Size**: 8, 16, 64
    - **Learning Rate**: 5e-4, 2e-5, 5e-5
    - **Epochs**: 5, 7, 10
    - **Warmup Steps**: 0, 500, 1000
  - Finally selected configuration:
    - Learning rate: 5e-4 (LoRA), 5e-5 (standard)
    - Batch Size: 16
    - Epochs: 7 (sufficient for convergence without overfitting)
    - Warmup Steps: 500 (gradual learning rate increase for stability)
    - Weight Decay: 0.01 (prevent overfitting)
  - Early stopping was implemented with a patience of 3 epochs, using macro-F1 as the metric to select the best model checkpoint.

- **Training Setup**:
  - Evaluation Strategy: Per epoch, with the best model loaded at the end based on ICM.
  - Logging: Every 10 steps for monitoring training dynamics.

This approach ensured robust fine-tuning, with LoRA providing an efficient alternative to full parameter updates.

## 4. Evaluation Measures and Results

The models were evaluated using two primary metrics to assess performance on the multi-label classification task:

- **Macro-F1 Score**:
  - Computed as the unweighted average of F1 scores across all labels.
  - Suitable for imbalanced datasets, ensuring equal importance to all categories (e.g., rare labels like "SEXUAL-VIOLENCE").

- **ICM**:
  - Measures the similarity between predicted and true label sets, penalizing incorrect or missing labels.

- **Training Time**:
  - Recorded as seconds per epoch to compare computational efficiency across models and methods.

### Results

The results for each language, model, and fine-tuning method are summarized below, based on the evaluation on the development set.

#### English Results
- **Fine-tuning**:
  - BERT: ICM: 0.0132, Macro-F1: 0.6746, Time/Epoch: 38.91
  - RoBERTa: ICM: 0.3494, Macro-F1: 0.7104, Time/Epoch: 20.65
- **LoRA**:
  - BERT: ICM: 0.1863, Macro-F1: 0.7060, Time/Epoch: 24.24
  - RoBERTa: ICM: 0.3129, Macro-F1: 0.7201, Time/Epoch: 32.36

#### Spanish Results
- **Fine-tuning**:
  - BERT (BETO): ICM: 0.1699, Macro-F1: 0.6915, Time/Epoch: 21.11
  - RoBERTa (BNE): ICM: -0.0227, Macro-F1: 0.6737, Time/Epoch: 21.38
- **LoRA**:
  - BERT (BETO): ICM: 0.1685, Macro-F1: 0.7097, Time/Epoch: 24.52
  - RoBERTa (BNE): ICM: -0.0601, Macro-F1: 0.6965, Time/Epoch: 25.08


## 5. Relevant Visualizations and Tables

### Visualizations

Three bar plots were generated to compare model performance and efficiency (included in source code file):

1. **Macro-F1 Scores**:
   - Compares F1 scores across models_languages and methods
   - Highlights RoBERTa’s superiority and the minimal performance drop with LoRA.

2. **ICM Scores**:
   - Shows ICM scores, reinforcing RoBERTa’s lead and LoRA’s competitiveness.

3. **Training Time per Epoch**:
   - Training time is similar for LoRA and traditional fine-tuning

The plots use seaborn’s barplot with `Model and Language` on the x-axis and `Method` as hue.

### Table


| Language   | Method      | Model   |     ICM |   Macro-F1 |   Time/Epoch (s) |
|:-----------|:------------|:--------|--------:|-----------:|-------------:|
| English    | Fine-tuning | BERT    |  0.0132 |     0.6746 |        38.91 |
| English    | Fine-tuning | RoBERTa |  0.3495 |     0.7105 |        20.65 |
| English    | LoRA        | BERT    |  0.1863 |     0.7061 |        24.24 |
| English    | LoRA        | RoBERTa |  0.3129 |     0.7201 |        23.11 |
| Spanish    | Fine-tuning | BERT    |  0.1699 |     0.6916 |        21.11 |
| Spanish    | Fine-tuning | RoBERTa | -0.0227 |     0.6737 |        21.38 |
| Spanish    | LoRA        | BERT    |  0.1685 |     0.7097 |        24.52 |
| Spanish    | LoRA        | RoBERTa | -0.0601 |     0.6965 |        25.08 |

This table confirms RoBERTa’s edge in performance in most of cases.

## 6. Conclusion


- **Preprocessing**: Standardizing inputs improved model focus on relevant content.
- **Model Selection**: RoBERTa-based models outperformed BERT-based models in most of the cases.
- **Fine-tuning**: LoRA doesn't provide great advantage in computational speed, but provides better results in some of the cases
- **Evaluation**: Macro-F1 and ICM metrics captured both overall and hierarchical performance, with visualizations and tables clarifying model comparisons.
