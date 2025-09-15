**Authors and affiliations**

<div style="font-size: larger;">
Anna Ouradova<sup>1,\*</sup>, Giulio Ferrero<sup>2,3,\*</sup>, Miriam Bratova<sup>4</sup>, Nikola Daskova<sup>4</sup>, Alena Bohdanecka<sup>4,5</sup>, Klara Dohnalova<sup>6</sup>, Marie Heczkova<sup>4</sup>, Karel Chalupsky<sup>6</sup>, Maria Kralova<sup>7,8</sup>, Marek Kuzma<sup>9</sup>, István Modos<sup>4</sup>, Filip Tichanek<sup>4</sup>, Lucie Najmanova<sup>9</sup>, Barbara Pardini<sup>10</sup>, Helena Pelantová<sup>9</sup>, Sonia Tarallo<sup>10</sup>, Petra Videnska<sup>11</sup>, Jan Gojda<sup>1,#</sup>, Alessio Naccarati<sup>10,#</sup>, Monika Cahova<sup>4,#</sup> 
</div>

<br>

<sup>*</sup> These authors have contributed equally to this work and share first authorship   
<sup>#</sup> These authors have contributed equally to this work and share last authorship   

<sup>1</sup> Department of Internal Medicine, Kralovske Vinohrady University Hospital and Third Faculty of Medicine, Charles University, Prague, Czech Republic     
<sup>2</sup> Department of  Clinical and Biological Sciences, University of Turin, Turin, Italy    
<sup>3</sup> Department of Computer Science, University of Turin, Turin, Italy      
<sup>4</sup> Institute for Clinical and Experimental Medicine, Prague, Czech Republic   
<sup>5</sup> First Faculty of Medicine, Charles University, Prague, Czech Republic   
<sup>6</sup> Czech Centre for Phenogenomics, Institute of Molecular Genetics of the Czech Academy of Sciences, Prague, Czech Republic   
<sup>7</sup> Ambis University, Department of Economics and Management, Prague, Czech Republic      
<sup>8</sup> Department of Informatics, Brno University of Technology, Brno, Czech Republic        
<sup>9</sup> Institute of Microbiology of the Czech Academy of Sciences, Prague, Czech Republic   
<sup>10</sup> Italian Institute for Genomic Medicine (IIGM), c/o IRCCS Candiolo, Turin, Italy    
<sup>11</sup> Mendel University, Department of Chemistry and Biochemistry, Brno, Czech Republic

---------------------------------------------------------------------------------------------------

This is a statistical report of the study *A vegan diet signature from a multi-omics study on different European populations is related to favorable metabolic outcomes* that is currenlty *under review*

When using this code or data, cite the original publication:

> TO BE ADDED

BibTex citation for the original publication:

> TO BE ADDED

---------------------------------------------------------------------------------------------------

Original [GitHub repository](https://github.com/filip-tichanek/ItCzVegans): https://github.com/filip-tichanek/ItCzVegans

Statistical **reports** can be found on the [reports hub](https://filip-tichanek.github.io/ItCzVegans/).

Data analysis is described in detail in the [statistical methods](https://filip-tichanek.github.io/ItCzVegans/html_reports/478_code04_methods.html) report.

----------------------------------------------------------------------------------------------------

# Introduction

This project explores potential signatures of a vegan diet across the microbiome, metabolome, and lipidome. We used data from healthy vegan and omnivorous human subjects in two countries (Czech Republic and Italy), with subjects grouped by `Country` and `Diet`, resulting in four distinct groups.

To assess the generalizability of these findings, we validated our results with an independent cohort from the Czech Republic for external validation.


## Statistical Methods

The statistical modeling approach is described in detail in [this report](https://filip-tichanek.github.io/ItCzVegans/html_reports/478_code04_methods.html). Briefly, the methods used included:

- **Multivariate analysis**: We conducted multivariate analyses (PERMANOVA, PCA, correlation analyses) to explore the effects of `diet`, `country`, and their possible interaction (`diet : country`) on the microbiome, lipidome, and metabolome compositions in an integrative manner. This part of the analysis is not available on the GitHub page, but the code will be provided upon request.

- **Linear models**: Linear models were applied to estimate the effects of `diet`, `country`, and their interaction (`diet:country`) on individual lipids, metabolites, bacterial taxa and pathways ("features"). Features that significantly differed between diet groups (based on the estimated average conditional effect of diet across both countries, adjusted for multiple comparisons with FDR < 0.05) were further examined in the independent external validation cohort to assess whether these associations were reproducible.
Next, we fit linear models restricted to vegan participants to test whether omics profiles varied with the duration of vegan diet. Fixed-effect predictors were diet duration (per 10 years), country, their interaction, and age (included due to correlation with diet duration).

- **Predictive models (elastic net)**:  We employed elastic net logistic regression (via the `glmnet` R package) to predict vegan status based on metabolome, lipidome, microbiome and pathways data (one model per dataset; four models in total). We considered three combinations of Lasso and Ridge penalties (alpha = 0, 0.2, 0.4). For each alpha, we selected the penalty strength (λ<sub>1se</sub>) using 10-fold cross-validation. This value corresponds to the most regularized model whose performance was within one standard error of the minimum deviance. The alpha–lambda pair with the lowest deviance was chosen to fit the final model, whose coefficients are reported.   
To estimate model performance, we repeated the full modeling procedure (including hyperparameter tuning) 500 times on bootstrap resamples of the training data. In each iteration, the model was trained on the resampled data and evaluated on the out-of-bag subjects (i.e., those not included in the training set in that iteration). The mean, and 2.5th, and 97.5th percentiles of the resulting ROC-AUC values represent the estimated out-of-sample AUC and its 95% confidence interval.   
Finally, the final model was applied to an independent validation cohort to generate predicted probabilities of vegan status. These probabilities were then used to assess external discrimination between diet groups (ROC-AUC in the independent validation cohort).
The elastic net models were not intended for practical prediction, but to quantify the strength of the signal separating the dietary groups, with its uncertainty, by using all features of a given dataset jointly. It also offered a complementary perspective on which features are most clearly associated with diet



