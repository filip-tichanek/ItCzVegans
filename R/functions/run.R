#' run Function
#'
#' This function loads or runs and saves (if not previously done) a model, table,
#' or any computationally intensive task. It's designed to avoid redundant computations
#' by reusing previously saved results when possible.
#'
#' @param expr An expression representing the computationally intensive task to be executed.
#' @param path A string specifying the file path where the result should be saved or loaded from.
#' @param reuse A logical flag indicating whether to attempt reusing saved results to avoid recomputation.
#' @return The result of evaluating `expr`. If `reuse` is TRUE and a saved result exists, 
#'         that result is returned; otherwise, `expr` is evaluated.
#' @examples
#' # Assuming lm_result.Rds does not exist, this will compute the linear model and save it.
#' run(lm(mpg ~ cyl, data = mtcars), path = "lm_result", reuse = TRUE)

run <- function(expr, path, reuse = TRUE) {
  fit <- NULL
  if (reuse) {
    path <- paste0(path, ".Rds")
    fit <- suppressWarnings(try(readRDS(path), silent = TRUE))
    if (inherits(fit, "try-error")) {
      fit <- NULL
    }
  }
  if (is.null(fit)) {
    fit <- eval(substitute(expr))
    if (reuse && !is.null(path) && nzchar(path)) {
      saveRDS(fit, file = path)
    }
  }
  return(fit)
}