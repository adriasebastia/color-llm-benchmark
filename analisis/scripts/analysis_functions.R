# Funcions base per l'analisi del benchmark.
# Si algo crece mucho, mejor seguir separandolo aqui y no en el notebook.

clean_dataset <- function(data) {
  if (!inherits(data, "data.frame")) {
    stop("clean_dataset espera un data.frame o tibble.")
  }

  data |>
    dplyr::mutate(dplyr::across(where(is.character), stringr::str_squish)) |>
    dplyr::distinct()
}

extract_main_color <- function(image_path) {
  if (!file.exists(image_path)) {
    stop(paste("No se ha encontrado la imagen:", image_path))
  }

  warning("extract_main_color es una estructura base: falta definir el metodo exacto.")
  NA_character_
}

evaluate_predictions <- function(data,
                                 expected_col = "expected_color",
                                 predicted_col = "predicted_color") {
  needed_cols <- c(expected_col, predicted_col)
  missing_cols <- setdiff(needed_cols, names(data))

  if (length(missing_cols) > 0) {
    stop(paste("Faltan columnas:", paste(missing_cols, collapse = ", ")))
  }

  data |>
    dplyr::mutate(correct = .data[[expected_col]] == .data[[predicted_col]]) |>
    dplyr::summarise(
      total = dplyr::n(),
      correct = sum(correct, na.rm = TRUE),
      accuracy = correct / total
    )
}
