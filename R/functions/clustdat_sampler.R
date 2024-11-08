#' Clustered Data Sampler
#'
#' This function generates resampled datasets from clustered data using various sampling methods.
#'
#' @param data A data frame containing the dataset to be sampled.
#' @param clust_id A string specifying the column name in `data` that identifies the clusters.
#' @param outcome A string specifying the column name in `data` that represents the outcome variable.
#' @param seed An optional integer for setting the random seed to ensure reproducibility.
#' @param sample_method A string specifying the sampling method. Options are 'boot' for 
#' common bootstrap, 'oos_boot' for out-of-sample bootstrap (suitable for evaluating out-of-sample
#' performance of a predictive model) and 'cv' for k-fold cross-validation. 
#' Default is 'boot'.
#' @param N An integer specifying the number of bootstrap samples or out-of-sample bootstrap 
#' samples to generate. Default is 10.
#' @param k An integer specifying the number of folds for cross-validation. Default is 10.
#'
#' @return Depending on `sample_method`, the function returns:
#' \itemize{
#'   \item If `sample_method` is 'boot', a list of `N` bootstrap samples.
#'   \item Otherwise, a list containing two lists: the training (resampled for 'oos_boot' method)
#'    dataset and the out-of-bag (validation) datasets (observation not included to training set).
#' }
#'
#' @examples
#' \dontrun{
#' # Example usage with bootstrap method
#' sampled_data <- clustdat_sampler(data, clust_id = "cluster", outcome = "outcome", seed = 123,
#' sample_method = 'boot', N = 10)
#'
#' # Example usage with out-of-sample bootstrap method
#' sampled_data <- clustdat_sampler(data, clust_id = "cluster", outcome = "outcome", seed = 123,
#' sample_method = 'oos_boot', N = 10)
#'
#' # Example usage with k-fold cross-validation
#' sampled_data <- clustdat_sampler(data, clust_id = "cluster", outcome = "outcome", seed = 123,
#' sample_method = 'kfold', k = 5)
#' }
#'
#' @export

clustdat_sampler <- function(
    data,
    clust_id,
    outcome,
    seed = NULL,
    sample_method = "boot",
    N = 10,
    k = 10) {
  if (!is.null(seed)) {
    set.seed(seed)
  }

  data <- data %>%
    dplyr::mutate(obs_id = as.character(1:nrow(data)))

  if (colnames(data[clust_id]) != "id") {
    data <- data %>%
      dplyr::mutate(id = data[[clust_id]]) %>%
      dplyr::select(-dplyr::all_of(clust_id))
  }

  if (colnames(data[outcome]) != "outcome") {
    data <- data %>%
      dplyr::mutate(outcome = data[[outcome]]) %>%
      dplyr::select(-dplyr::all_of(outcome))
  }

  data <- data %>%
    dplyr::mutate(obs_id = as.character(1:nrow(data))) %>%
    dplyr::select(obs_id, id, dplyr::everything())


  if (sample_method == "boot") {
    reset <- list()

    for (i in 1:N) {
      tmp <- data.frame(id = sample(unique(data$id),
        length(unique(data$id)),
        replace = TRUE
      ))

      tmp <- tmp %>%
        mutate(id_sec = factor(1:nrow(tmp))) %>%
        left_join(data,
          by = "id",
          relationship = "many-to-many"
        )

      reset[[i]] <- tmp
    }

    return(reset)
  } else if (sample_method == "oos_boot") {
    train_data <- list()
    valid_data <- list()

    for (i in 1:N) {
      repeat {
        train <- data.frame(id = sample(unique(data$id),
          length(unique(data$id)),
          replace = TRUE
        ))

        temp_train <- train %>%
          dplyr::left_join(
            data, 
            by = "id", 
            relationship = "many-to-many"
          )

        temp_valid <- data.frame(
          obs_id = data[!data$obs_id %in% temp_train$obs_id, ]$obs_id
        ) %>%
          dplyr::left_join(data, by = "obs_id")

        if (!mean(temp_train$outcome) %in% c(0, 1) & !mean(temp_valid$outcome) %in% c(0, 1)) {
          train_data[[i]] <- temp_train
          valid_data[[i]] <- temp_valid
          {
            break
          }
        }
      }
    }

    return(list(train_data, valid_data))
    
  } else if (sample_method == "cv") {
    
    unique_ids <- unique(data$id)

    id_groups <- sample(rep(1:k, length.out = length(unique_ids)))

    train_data <- list()
    valid_data <- list()

    for (i in 1:k) {
      ids_include <- unique_ids[id_groups != i]
      ids_not_include <- unique_ids[id_groups == i]

      train_data[[i]] <- data %>%
        dplyr::filter(id %in% ids_include)

      valid_data[[i]] <- data %>%
        dplyr::filter(id %in% ids_not_include)
    }

    return(list(train_data, valid_data))
  } else {
    stop("`sample_method` is not correctly specified. Available options: `boot` for classical bootstrap, `oos_boot` for out-of-sample bootstrap, and `cv` for cross-validation", call. = FALSE)
  }
}
