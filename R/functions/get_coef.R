get_coef <- function(original_data,
                     glmnet_model) {
  
  betas_val <- data.frame(
    predictor = row.names(
      glmnet_model$beta
    ),
    beta_scaled = glmnet_model$betas[, 1]
  ) %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      SD = sd(original_data[[predictor]], na.rm = TRUE),
      mean = mean(original_data[[predictor]], na.rm = TRUE)
    ) %>%
    dplyr::ungroup() %>%
    dplyr::mutate(
      beta_OrigScale = beta_scaled * SD * 2
    ) %>%
    dplyr::select(
      predictor,
      beta_scaled,
      beta_OrigScale,
      mean,
      SD
    )
  
  return(betas_val)
}