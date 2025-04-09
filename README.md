**Authors and affiliations**

<div style="font-size: larger;">
Monika Cahova<sup>1,*</sup>, Anna Ouradova<sup>2,*</sup>, Giulio Ferrero<sup>3,4,*</sup>, Miriam Bratova<sup>1</sup>, Nikola Daskova<sup>1</sup>, Klara Dohnalova<sup>5</sup>, Marie Heczkova<sup>1</sup>, Karel Chalupsky<sup>5</sup>, Maria Kralova<sup>6,7</sup>, Marek Kuzma<sup>8</sup>, Filip Tichanek<sup>1</sup>, Lucie Najmanova<sup>8</sup>, Barbara Pardini<sup>10</sup>, Helena Pelantová<sup>8</sup>, Radislav Sedlacek<sup>5</sup>, Sonia Tarallo<sup>9</sup>, Petra Videnska<sup>10</sup>, Jan Gojda<sup>2,#</sup>, Alessio Naccarati<sup>9,#</sup>
</div>

<br>

<sup>*</sup> These authors have contributed equally to this work and share first authorship   
<sup>#</sup> These authors have contributed equally to this work and share last authorship   

<sup>1</sup> Institute for Clinical and Experimental Medicine, Prague, Czech Republic     
<sup>2</sup> Department of Internal Medicine, Kralovske Vinohrady University Hospital and Third Faculty of Medicine, Charles University, Prague, Czech Republic   
<sup>3</sup> Department of Clinical and Biological Sciences, University of Turin, Turin, Italy      
<sup>4</sup> Department of Computer Science, University of Turin, Turin, Italy   
<sup>5</sup> Czech Centre for Phenogenomics, Institute of Molecular Genetics of the Czech Academy of Sciences, Prague, Czech Republic   
<sup>6</sup> Ambis University, Department of Economics and Management, Prague, Czech Republic   
<sup>7</sup> Department of Applied Mathematics and Computer Science, Masaryk University, Brno, Czech Republic   
<sup>8</sup> Institute of Microbiology of the Czech Academy of Sciences, Prague, Czech Republic        
<sup>9</sup> Italian Institute for Genomic Medicine (IIGM), c/o IRCCS Candiolo, Turin, Italy   
<sup>10</sup> Mendel University, Department of Chemistry and Biochemistry, Brno, Czech Republic

---------------------------------------------------------------------------------------------------

This is a statistical report of the study *Vegan-specific signature implies healthier metabolic profile: findings from diet-related multi-omics observational study based on different European populations* that is *under review* in [Nature Communications](https://www.nature.com/ncomms/)

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

- **Linear models**: Linear models were applied to estimate the effects of `diet`, `country`, and their interaction (`diet:country`) on individual lipids, metabolites, bacterial taxa and pathways ("features"). Features that significantly differed between diet groups (based on the estimated average effect of diet across both countries, adjusted for multiple comparisons with FDR < 0.05) were further examined in the independent validation cohort to assess whether these associations were reproducible.

- **Predictive models (elastic net)**: We employed elastic net (regularized) logistic regression to predict vegan status based on metabolome, lipidome, microbiome and pathways (one predictive model per dataset, i.e.,four elastic net models in total). These models were internally validated using out-of-bag bootstrap resampling. The discriminatory power of each model to differentiate between diet groups was evaluated using the out-of-sample (optimism-corrected) area under the receiver operating characteristic curve (ROC-AUC). The models trained on the training data were then used to estimate the predicted probability that a given subject is vegan in an indepedent validation cohort. This predicted probability was subsequently used as a variable to discriminate between diet groups for external validation.


