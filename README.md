# DeepInteraction
Using deep learning to predict complex traits using genotypes, utilizing non-linearity of a MLP. NID is an algorithm developed by M. Tsang, that is subsequently applied on a trained MLP to predict genetically mediated complex traits, to discover epistatic interactions between single-nucleotide polymorphisms (SNPs).
See M. Tsangs' github: https://github.com/mtsang/neural-interaction-detection


A quick overview of the architecture of the neural network (MLP) used in this project.


<img width="766" alt="Screenshot 2025-03-11 at 9 41 44 PM" src="https://github.com/user-attachments/assets/7ac787d2-8ad7-498f-8ef8-c38dfd8ee186" />
<img width="554" alt="Screenshot 2025-03-11 at 9 42 10 PM" src="https://github.com/user-attachments/assets/6953b593-8ce0-4374-8bb8-373101f441ec" />



To account for imbalanced classes I implemented my own loss function in pytorch by relying on matthew's correlation coefficient, utilizing sklean (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.matthews_corrcoef.html). See formula here

<img width="416" alt="Screenshot 2025-03-11 at 9 43 31 PM" src="https://github.com/user-attachments/assets/e2fd39bc-ad5d-4c00-947c-61daa07c9a37" />

Where TP is the true positive rate, TN true negatives, FP false positives and FN false negatives, i.e. accounting for all possible outcomes/errors. Matthew's correlation coefficient was only used during the validation step of the fitting process.


Neural interaction detection is based on Tsang et al. 2017 (https://openreview.net/pdf?id=ByOfBggRZ) (see github above).

<img width="764" alt="Screenshot 2025-03-11 at 9 46 46 PM" src="https://github.com/user-attachments/assets/b6caa72d-a00e-44ee-b41b-e7d3d7542631" />


In order to validate the ability of NID to discover epistatic interactions we had to figure out a way to estimate the rank of an interaction in a large list of interactions (ca. 22 million rows) and a normal sorting procedure was not feasible, since the model had to be run 100 times to create a distribution of ranks. We relied on an algorithm similar to bubble sort developed by Peter Sackett from DTU (Technical University of Denmark).


<img width="419" alt="Screenshot 2025-03-11 at 9 50 35 PM" src="https://github.com/user-attachments/assets/2f1cec38-5f76-48a7-95f8-cfd06277d299" />



Recovery of simulated interactions with comparison with shuffle phenotypes (null distribution). We see clearly that the ground truth interactions populates the lower ranking interaciotns (i.e. strongest interactions) quantified by NID and recovered by Sackett's algorithm. This is the rank over 100 model runs.

<img width="903" alt="Screenshot 2025-03-11 at 9 53 38 PM" src="https://github.com/user-attachments/assets/57495de5-c92c-492b-b80b-a4cca3f6cd20" />

