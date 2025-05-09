
### Preprocessing:
- removal of stopwords, special characters
- lowercasing
- tokenization

### Text representations:
- traditional - TF-IDF
- static word-embeddings

### Classification models:
- Decision Tree
- Multilayer Perceptron
- Random forest - Ensemble of Decision Trees
- Stacking - Decision Tree + Multilayer Perceptron + SVM and as final model Linear Regression


### Hyperparameters:
1. Random forest
    - number of trees - considered values -> 50, 100, 200 -> more than 50 doesn't change result and generally Random Forest on this dataset gives results similar to single Decision Tree
2. Multilayer Perceptron
    - number of hidden layers - considered values -> 2, 4, 6, 8 -> more than 4 doesn't improve results
    - size of hidden layer - considered values -> 32, 64, 128, 256 -> 256 doesn't improve results anymore
    - activation functions - relu gives best results
    - batch size - considered values -> 8, 16, 32, 64
3. Stacking
    - Multilayer Perceptron
        - size of hidden layer - considered values -> 50, 100, 200
        - max_iter - considered values -> 10, 15, 50 -> 50 takes too long
    - SVM
        - kernels


### Best results:
1. Task 1 English - `0.746` with usage of:
    - removal of stopwords, special characters
    - lowercasing
    - TF-IDF
    - Stacking - Decision Tree + Multilayer Perceptron + SVM and as final model Linear Regression

2. Task 2 English - `0.492` with usage of:
    - removal of stopwords, special characters
    - lowercasing
    - tokenization
    - static word-embeddings
    - Multilayer Perceptron

3. Task 1 Spanish - `0.748` with usage of:
    - removal of stopwords, special characters
    - lowercasing
    - TF-IDF
    - Random Forest

4. Task 2 Spanish - `0.430` with usage of:
    - removal of stopwords, special characters
    - lowercasing
    - TF-IDF
    - Decision Tree
