#' clean Function
#' 
#' Cleans a string by performing several transformations to make it more suitable for regex mining.
#' This function transliterates characters to ASCII, converts the string to lowercase, 
#' and removes diacritics, punctuation, spaces, and specific symbols such as "+" and "-".
#'
#' @param string_character A character string to be cleaned.
#' @return The cleaned text as a lower-case string without spaces, punctuation, or symbols.
#' @examples
#' clean("Example+ text - with, punctuation!")
#' @importFrom stringi stri_trans_general
#' @importFrom stringr str_replace_all str_to_lower

clean <- function(string_character) {
  require(stringi)
  require(stringr)
  
  string_character %>%
    stringi::stri_trans_general("Latin-ASCII") %>% 
    stringr::str_replace_all("[[:punct:]/\\-]+", "") %>%  
    stringr::str_replace_all("\\s+", "") %>% 
    stringr::str_replace_all('\\+', "") %>% 
    stringr::str_to_lower()
}