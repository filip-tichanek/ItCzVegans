# Initiation file


## Package upload
if (TRUE) {rm(list = ls() )}
if (TRUE) {
  suppressWarnings(suppressMessages({
    ### PASSWORD
    #library(rstudioapi)
    ### DATA: obtain, wrangle and explore
    library(RJDBC)
    library(openxlsx)
    library(readxl)
    library(tidyverse)
    library(purrr)
    library(lubridate)
    library(stringi)
    library(janitor)
    library(compositions)
    ### TABLES: create and modify
    library(gtsummary)
    library(flextable)
    library(kableExtra)
    library(sjPlot)
    ### PLOTS
    library(ggpubr)
    library(cowplot)
    library(ggdist)
    ### STATS
    library(glmmTMB)
    library(rms)
    library(brms)
    library(emmeans)
    library(car)
    library(arm)
    library(pROC)
    library(glmnet)
    library(MicrobiomeStat)
  }))
}

## Functions


### set functions clashes
select <- dplyr::select
rename <- dplyr::rename
mutate <- dplyr::mutate
recode <- dplyr::recode
summarise <- dplyr::summarise
count <- dplyr::count

### loading custom functions
invisible(
  lapply(
    list.files(
      "functions", pattern = "\\.R$", full.names = TRUE), 
    source)
)


## Create folders
folders <- c("data", 
             "data/db_history", 
             "gitignore",
             "gitignore/data",
             "gitignore/run",
             "gitignore/figures",
             "gitignore/html_reports")

invisible(
  lapply(
    folders, function(x) if (!dir.exists(x)) 
      dir.create(x, recursive = TRUE)
  )
)


## Set seed
set.seed(16)