#' compare_predmod
#'
#' This function compares two predictive models based on a specified measure.
#'
#' @param model_simpler A list containing the simpler model's performance data.
#' @param model_complex A list containing the complex model's performance data.
#' @param measure A string specifying the performance measure to compare. Default is
#' 'auc_resamp_validation' (currently the only option)
#'
#' @return A data frame with the following columns:
#' \itemize{
#'   \item \code{auc_dif} The mean difference in the specified measure.
#'   \item \code{CI_L} The lower bound of the confidence interval for the difference.
#'   \item \code{CI_U} The upper bound of the confidence interval for the difference.
#'   \item \code{P} The proportion of times the complex model outperforms the simpler model.
#' }
#'
#' @examples
#' \dontrun{
#' # Example usage
#' comparison <- compare_predmod(model_simpler, model_complex, measure = 'auc_resamp_validation')
#' }
#'
#' @export


compare_predmod <- function(model_simpler, 
                            model_complex, 
                            measure = 'auc_resamp_validation'){
  
  model_simpler <- model_simpler$valid_performances
  model_complex <- model_complex$valid_performances
  
  min_nrow <- min(c(nrow(model_simpler), nrow(model_complex)))
  
  model_simpler <- model_simpler[1:min_nrow,]
  model_complex <- model_complex[1:min_nrow,]
  
  tr <- c(model_complex[[measure]] - model_simpler[[measure]])
  
  result <- data.frame(
    auc_dif = mean(tr),
    CI_L = quantile(tr, probs = 0.025),
    CI_U = quantile(tr, probs = 0.975),
    P = (min(c(length(tr[tr<0]), length(tr[tr>0]))) + 0.5)/(0.5 + length(tr)*0.5)
  )
  
  return(t(result))
  
}