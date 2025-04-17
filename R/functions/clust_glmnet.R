#' clustered_glmnet
#'
#' This function fits a GLMNET model to clustered data with different sampling methods.
#'
#' @param data A data frame containing the dataset to be sampled.
#' @param outcome A string specifying the column name in `data` representing the outcome variable.
#' @param clust_id A string specifying the column name in `data` identifying the clusters.
#' @param sample_method A string specifying the sampling method. Options are 'boot' for bootstrap,
#' 'oos_boot' for the bootstrap specifically designed for predictive model validation 
#' (standard cluster bootstrap to get training data and rest data for validation),
#'  and 'cv' for cross-validation. Default is 'oos_boot'.  
#' @param sampled_data A data frame containing already acquired resamples
#' @param N An integer specifying the number of bootstrap or atypical bootstrap samples to
#' generate. Default is 10.
#' @param k An integer specifying the number of folds for cross-validation. Default is 10.
#' @param alphas A numeric vector of alpha values to be tested in GLMNET. 
#' Default is seq(0, 1, by = 0.2).
#' @param family A string specifying the family for the GLMNET model. 
#' Default is 'binomial' (currently the only option).
#' @param seed An optional integer for setting the random seed to ensure reproducibility. 
#' Default is 123.
#'
#' @return A list containing the following elements:
#' \itemize{
#'   \item \code{model_summary} A data frame with model summary statistics.
#'   \item \code{valid_performances} A data frame with validation performances.
#'   \item \code{predictions} A data frame with predictions.
#'   \item \code{betas} A matrix with model coefficients.
#'   \item \code{plot} A ggplot object with the calibration plot.
#' }
#'
#' @examples
#' \dontrun{
#' # Example usage with default parameters
#' result <- clustered_glmnet(data, outcome = "outcome", clust_id = "cluster")
#' }
#'
#' @export

clust_glmnet <- function(
    data,
    outcome,
    clust_id,
    sample_method = "oos_boot",
    sampled_data = NULL,
    N = 10,
    k = 10,
    alphas = seq(0, 1, by = 0.5),
    family = "binomial",
    seed = 123,
    standardize = FALSE) {

  set.seed(seed)

  if (!sample_method %in% c("oos_boot", "boot", "cv")) {
    stop("`sample_method` is not correctly specified. Available options: `boot` for classical optimisms correction with bootstrap as suggested by Effron, `oos_boot` for a simple out-of-sample bootstrap, and `cv` for cross-validation", call. = FALSE)
  }

  ## where to save relevant information
  auc_resamp_test <- vector("double", N)
  accuracy_resamp_test <- vector("double", N)
  auc_resamp_validation <- vector("double", N)
  accuracy_resamp_validation <- vector("double", N)
  predictions <- vector("list", N)


  ## original data in a matrix form
  original_outcome <- base::as.matrix(data[[outcome]])
  original_predictors <- data %>%
    dplyr::select(
      -dplyr::all_of(
        c(outcome, clust_id)
      )
    ) %>%
    as.matrix()



  # glmnet on original sample ----

  ## optimize lambda and alpha
  lamb_1se <- vector("double", length(alphas))
  alpha <- vector("double", length(alphas))
  deviance <- vector("double", length(alphas))

  for (a in seq_along(alphas)) {
    tr <- glmnet::cv.glmnet(
      x = original_predictors,
      y = original_outcome,
      alpha = alphas[a],
      family = family,
      type.measure = "deviance",
      standardize = standardize
    )
    lamb_1se[a] <- tr[["lambda.1se"]]

    alpha[a] <- alphas[a]

    deviance[a] <- tr$cvm[
      which(
        tr$lambda == tr[["lambda.1se"]]
      )
    ]
  }

  optim_par <- data.frame(
    lamb_1se,
    alpha,
    deviance
  ) %>%
    dplyr::arrange(deviance)


  ## fit with optimized hyperparameters
  fit <- glmnet::glmnet(
    x = original_predictors,
    y = original_outcome,
    alpha = optim_par$alpha[1],
    lambda = optim_par$lamb_1se[1],
    family = family,
    standardize = standardize
  )


  ## get predictions and performance
  prediction <- data.frame(
    predicted_orig = as.numeric(
      predict(
        fit,
        newx = original_predictors
      )
    ),
    outcome = original_outcome
  )

  fitted <- data.frame(
    alpha = optim_par$alpha[1],
    lambda = optim_par$lamb_1se[1],
    auc = pROC::roc(
      outcome ~ predicted_orig,
      data = prediction,
      direction = "<",
      levels = c(0, 1)
    )$auc,
    accuracy = mean(
      ifelse(
        prediction$predicted_orig > 0,
        1,
        0
      ) == prediction$outcome
    )
  )



  # glmnet on simulated data ----


  ## simulated data

  if (is.null(sampled_data)) {
    
    set.seed(seed)
      
    sampled_data <- clustdat_sampler(
      data = data,
      clust_id = clust_id,
      outcome = outcome,
      sample_method = sample_method,
      N = N,
      k = k,
      seed = seed
    )
  }

  niter <- ifelse(sample_method == 'cv', k, N)
  
  for (i in 1:niter) {
    if (sample_method == "boot") {
      sampled_outcome <- as.matrix(
        sampled_data[[i]]$outcome
      )

      sampled_predictors <- sampled_data[[i]] %>%
        dplyr::select(
          -outcome,
          -id,
          -id_sec,
          -obs_id
        ) %>%
        as.matrix()
    } else {
      sampled_outcome <- as.matrix(
        sampled_data[[1]][[i]]$outcome
      )

      sampled_predictors <- sampled_data[[1]][[i]] %>%
        dplyr::select(
          -outcome,
          -id,
          -obs_id
        ) %>%
        as.matrix()
    }


    ## re-optimize alpha and lambda
    lamb_1se <- vector("double", length(alphas))
    alpha <- vector("double", length(alphas))
    deviance <- vector("double", length(alphas))
    
    set.seed(seed)
    
    for (a in seq_along(alphas)) {
      
      tr <- glmnet::cv.glmnet(
        x = sampled_predictors,
        y = sampled_outcome,
        alpha = alphas[a],
        family = family,
        type.measure = "deviance",
        standardize = standardize
      )

      lamb_1se[a] <- tr[["lambda.1se"]]

      alpha[a] <- alphas[a]
      deviance[a] <- tr$cvm[
        which(
          tr$lambda == tr[["lambda.1se"]]
        )
      ]
    }

    optim_par <- data.frame(
      lamb_1se,
      alpha,
      deviance
    ) %>%
      dplyr::arrange(deviance)


    ## fit models with re-optimized hyperparameters
    sampled_fit <- glmnet::glmnet(
      sampled_predictors,
      sampled_outcome,
      alpha = optim_par$alpha[1],
      lambda = optim_par$lamb_1se[1],
      family = family,
      standardize = standardize
    )


    ## get predictions
    prediction_onSampled <- data.frame(
      predicted = as.numeric(
        predict(
          sampled_fit,
          newx = sampled_predictors
        )
      ),
      outcome = sampled_outcome
    )

    if (sample_method == "boot") {
      prediction_onValidation <- data.frame(
        predicted = as.numeric(
          predict(
            sampled_fit,
            newx = original_predictors
          )
        ),
        outcome = original_outcome,
        iteration = i
      )
    } else {
      valid_outcome <- as.matrix(
        sampled_data[[2]][[i]]$outcome
      )

      valid_predictors <- sampled_data[[2]][[i]] %>%
        dplyr::select(
          -outcome,
          -id,
          -obs_id
        ) %>%
        as.matrix()

      prediction_onValidation <- data.frame(
        predicted = as.numeric(
          predict(
            sampled_fit,
            newx = valid_predictors
          )
        ),
        outcome = valid_outcome,
        iteration = i
      )
    }


    ## record performance measures

    auc_resamp_test[i] <- pROC::roc(
      outcome ~ predicted,
      data = prediction_onSampled,
      direction = "<",
      levels = c(0, 1)
    )$auc

    accuracy_resamp_test[i] <- mean(
      ifelse(
        prediction_onSampled$predicted > 0, 1, 0
      ) == prediction_onSampled$outcome
    )

    auc_resamp_validation[i] <- pROC::roc(
      outcome ~ predicted,
      data = prediction_onValidation,
      direction = "<",
      levels = c(0, 1)
    )$auc

    accuracy_resamp_validation[i] <- mean(
      ifelse(
        prediction_onValidation$predicted > 0, 1, 0
      ) == prediction_onValidation$outcome
    )


    if (sample_method == "boot") {
      predictions[[i]] <- data.frame(
        prediction_onValidation,
        id_obs = data[[clust_id]]
      )
    } else {
      predictions[[i]] <- data.frame(
        prediction_onValidation,
        id_obs = sampled_data[[2]][[i]]$obs_id
      )
    }
  }

  ## connect predictions
  predictions <- bind_rows(predictions)


  predictions2 <- predictions %>%
    dplyr::group_by(id_obs) %>%
    dplyr::summarise(
      predicted = mean(predicted),
      outcome = mean(outcome)
    ) %>%
    dplyr::ungroup()


  # aggregate information ----

  valid_performances <- data.frame(
    auc_resamp_test,
    auc_resamp_validation,
    auc_optimism = auc_resamp_test - auc_resamp_validation,
    accuracy_resamp_test,
    accuracy_resamp_validation,
    accuracy_optimism = accuracy_resamp_test - accuracy_resamp_validation
  )

  betas <- coef(fit)

  if (sample_method == "boot") {
    model_summary <- fitted %>%
      mutate(
        auc_optimism_corrected = auc - mean(
          valid_performances$auc_optimism
        ),
        auc_optimism_corrected_CIL = auc - quantile(
          valid_performances$auc_optimism,
          probs = 0.975
        ),
        auc_optimism_corrected_CIU = auc - quantile(
          valid_performances$auc_optimism,
          probs = 0.025
        ),
        accuracy_optimism_corrected = accuracy - mean(
          valid_performances$accuracy_optimism
        ),
        accuracy_optimism_corrected_CIL = accuracy - quantile(
          valid_performances$accuracy_optimism,
          probs = 0.975
        ),
        accuracy_optimism_corrected_CIU = accuracy - quantile(
          valid_performances$accuracy_optimism,
          probs = 0.025
        )
      ) %>%
      dplyr::select(
        alpha, lambda,
        auc,
        auc_optimism_corrected:accuracy_optimism_corrected_CIU
      )
  } else if (sample_method == "oos_boot") {
    model_summary <- fitted %>%
      dplyr::mutate(
        auc_OutOfSample = mean(
          valid_performances$auc_resamp_validation
        ),
        auc_oos_CIL = quantile(
          valid_performances$auc_resamp_validation,
          probs = 0.025
        ),
        auc_oos_CIU = quantile(
          valid_performances$auc_resamp_validation,
          probs = 0.975
        ),
        accuracy_OutOfSample = mean(
          valid_performances$accuracy_resamp_validation
        ),
        accuracy_oos_CIL = quantile(
          valid_performances$accuracy_resamp_validation,
          probs = 0.025
        ),
        accuracy_oos_CIU = quantile(
          valid_performances$accuracy_resamp_validation,
          probs = 0.975
        )
      ) %>%
      dplyr::select(
        alpha, lambda,
        auc,
        auc_OutOfSample:auc_oos_CIU,
        accuracy,
        accuracy_OutOfSample:accuracy_oos_CIU
      )
  } else if (sample_method == "cv") {
    model_summary <- fitted %>%
      dplyr::mutate(
        auc_OutOfSample = mean(
          valid_performances$auc_resamp_validation
        ),
        accuracy_OutOfSample = mean(
          valid_performances$accuracy_resamp_validation
        )
      ) %>%
      dplyr::select(
        alpha, lambda,
        auc,
        auc_OutOfSample,
        accuracy,
        accuracy_OutOfSample
      )
  } else {
    stop("`sample_method` is not correctly specified. Available options: `boot` for classical bootstrap, `oos_boot` for out-of-sample bootstrap, and `cv` for cross-validation", call. = FALSE)
  }



  # calibration plot ----

  if (sample_method == "oos_boot") {
    set.seed(seed)
    
    calibration_plot <- suppressWarnings(
      predictions2 %>%
        dplyr::mutate(
          iteration = factor("A"),
          predicted = inv_logit(predicted)
        ) %>%
        ggplot2::ggplot(
          ggplot2::aes(
            x = predicted, 
            y = outcome, 
            group = iteration
            )
          ) +
        ggplot2::geom_smooth(
          data = predictions[factor(predictions$iteration) %in% factor(sample(1:N, 50)),],
          aes(
            x = inv_logit(predicted),
            y = outcome
          ),
          se = FALSE,
          color = "grey35",
          linewidth = 0.1,
          method = "loess",
          span =  1.6/log10(nrow(predictions2)),
          formula = 'y ~ x'
        ) +
        ggplot2::geom_smooth(
          method = "loess",
          se = TRUE,
          color = "red",
          fill = "red",
          alpha = 0.25,
          span = 1/log10(nrow(predictions2)),
          formula = 'y ~ x'
        ) +
        ggplot2::coord_cartesian(
          x = c(
            min(
              inv_logit(predictions$predicted)
            ),
            max(
              inv_logit(predictions$predicted)
            )
          ),
          y = c(0, 1)
        ) +
        ggplot2::geom_abline(
          slope = 1,
          intercept = 0,
          linewidth = 1,
          linetype = "dashed"
        ) +
        ggplot2::labs(
          x = "Prediction",
          y = "Outcome"
        )
    )
  }


  # define outputs ----
  
  if(sample_method == 'oos_boot'){
    
    return(list(
      model_summary = model_summary,
      valid_performances = valid_performances,
      predictions = prediction,
      betas = betas,
      plot = calibration_plot,
      fit = fit
    ))
    } else {
  
  return(list(
    model_summary = model_summary,
    valid_performances = valid_performances,
    predictions = prediction,
    betas = betas, 
    fit = fit
  ))
    }
}

