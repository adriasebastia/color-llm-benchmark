# Funcions per a l'analisi estadistica del benchmark de colors.
# La resposta principal es error_cromatic, el factor es model i la covariable es chroma.

clean_dataset <- function(data) {
  if (!inherits(data, "data.frame")) {
    stop("clean_dataset espera un data.frame o tibble.")
  }

  data |>
    dplyr::mutate(dplyr::across(where(is.character), stringr::str_squish)) |>
    dplyr::distinct()
}

prepare_benchmark_data <- function(data) {
  needed_cols <- c("image_name", "model", "status", "error_cromatic", "chroma")
  missing_cols <- setdiff(needed_cols, names(data))

  if (length(missing_cols) > 0) {
    stop(paste("Falten columnes:", paste(missing_cols, collapse = ", ")))
  }

  clean_dataset(data) |>
    dplyr::filter(.data$status == "ok") |>
    dplyr::mutate(
      error_cromatic = as.numeric(.data$error_cromatic),
      chroma = as.numeric(.data$chroma),
      model = as.character(.data$model)
    ) |>
    dplyr::filter(!is.na(.data$error_cromatic), !is.na(.data$chroma))
}

validate_paired_design <- function(data, expected_models = c("gpt-4o", "gpt-4o-mini")) {
  observed_models <- sort(unique(data$model))

  if (!setequal(observed_models, expected_models)) {
    stop(
      paste(
        "Models inesperats. Observats:",
        paste(observed_models, collapse = ", "),
        "| Esperats:",
        paste(expected_models, collapse = ", ")
      )
    )
  }

  by_image <- data |>
    dplyr::count(.data$image_name, .data$model, name = "rows_per_model") |>
    dplyr::group_by(.data$image_name) |>
    dplyr::summarise(
      n_models = dplyr::n_distinct(.data$model),
      total_rows = sum(.data$rows_per_model),
      max_rows_per_model = max(.data$rows_per_model),
      .groups = "drop"
    )

  bad_pairs <- by_image |>
    dplyr::filter(
      .data$n_models != length(expected_models) |
        .data$total_rows != length(expected_models) |
        .data$max_rows_per_model != 1
    )

  if (nrow(bad_pairs) > 0) {
    stop(paste("Hi ha imatges sense parella completa o amb duplicats:", nrow(bad_pairs)))
  }

  invisible(by_image)
}

model_error_key <- function(model) {
  dplyr::case_when(
    model == "gpt-4o" ~ "error_gpt_4o",
    model == "gpt-4o-mini" ~ "error_gpt_4o_mini",
    TRUE ~ paste0("error_", gsub("[^A-Za-z0-9]+", "_", model))
  )
}

create_paired_errors <- function(data) {
  validate_paired_design(data)

  paired <- data |>
    dplyr::mutate(model_key = model_error_key(.data$model)) |>
    dplyr::select("image_name", "chroma", "model_key", "error_cromatic") |>
    tidyr::pivot_wider(names_from = "model_key", values_from = "error_cromatic") |>
    dplyr::mutate(D = .data$error_gpt_4o_mini - .data$error_gpt_4o)

  if (any(is.na(paired$D))) {
    stop("La taula aparellada conte diferencies NA.")
  }

  paired
}

descriptive_by_model <- function(data) {
  data |>
    dplyr::group_by(.data$model) |>
    dplyr::summarise(
      n = dplyr::n(),
      mean = mean(.data$error_cromatic),
      median = median(.data$error_cromatic),
      sd = sd(.data$error_cromatic),
      min = min(.data$error_cromatic),
      max = max(.data$error_cromatic),
      .groups = "drop"
    )
}

paired_ttest_summary <- function(paired_data) {
  test <- t.test(paired_data$D)

  tibble::tibble(
    n = length(paired_data$D),
    mean_D = mean(paired_data$D),
    sd_D = sd(paired_data$D),
    conf_low = unname(test$conf.int[1]),
    conf_high = unname(test$conf.int[2]),
    t = unname(test$statistic),
    df = unname(test$parameter),
    p_value = test$p.value
  )
}

linear_model_summary <- function(model, label) {
  model_summary <- summary(model)
  coefficients <- model_summary$coefficients

  tibble::tibble(
    model = label,
    term = rownames(coefficients),
    estimate = coefficients[, "Estimate"],
    std_error = coefficients[, "Std. Error"],
    t = coefficients[, "t value"],
    p_value = coefficients[, "Pr(>|t|)"],
    r_squared = model_summary$r.squared,
    adj_r_squared = model_summary$adj.r.squared,
    residual_sd = model_summary$sigma
  )
}

fit_error_chroma_models <- function(data) {
  split(data, data$model) |>
    purrr::imap(~ lm(error_cromatic ~ chroma, data = .x))
}

summarise_error_chroma_models <- function(models) {
  purrr::imap_dfr(models, linear_model_summary)
}

plot_error_distribution <- function(data) {
  ggplot2::ggplot(data, ggplot2::aes(x = .data$error_cromatic, fill = .data$model)) +
    ggplot2::geom_histogram(alpha = 0.65, bins = 35, position = "identity") +
    ggplot2::facet_wrap(ggplot2::vars(.data$model), ncol = 1) +
    ggplot2::labs(x = "Error cromatic", y = "Nombre d'imatges", fill = "Model") +
    ggplot2::theme_minimal()
}

plot_error_boxplot <- function(data) {
  ggplot2::ggplot(data, ggplot2::aes(x = .data$model, y = .data$error_cromatic, fill = .data$model)) +
    ggplot2::geom_boxplot(alpha = 0.75, outlier_alpha = 0.35) +
    ggplot2::labs(x = "Model", y = "Error cromatic", fill = "Model") +
    ggplot2::theme_minimal()
}

plot_paired_difference_qq <- function(paired_data) {
  ggplot2::ggplot(paired_data, ggplot2::aes(sample = .data$D)) +
    ggplot2::stat_qq() +
    ggplot2::stat_qq_line() +
    ggplot2::labs(x = "Quantils teorics normals", y = "Quantils observats de D") +
    ggplot2::theme_minimal()
}

plot_paired_difference_histogram <- function(paired_data) {
  ggplot2::ggplot(paired_data, ggplot2::aes(x = .data$D)) +
    ggplot2::geom_histogram(bins = 35, fill = "#4C78A8", alpha = 0.8) +
    ggplot2::geom_vline(xintercept = 0, linetype = "dashed") +
    ggplot2::labs(x = "D = error gpt-4o-mini - error gpt-4o", y = "Nombre d'imatges") +
    ggplot2::theme_minimal()
}

plot_error_vs_chroma <- function(data) {
  ggplot2::ggplot(data, ggplot2::aes(x = .data$chroma, y = .data$error_cromatic, color = .data$model)) +
    ggplot2::geom_point(alpha = 0.35) +
    ggplot2::geom_smooth(method = "lm", se = TRUE) +
    ggplot2::labs(x = "Chroma", y = "Error cromatic", color = "Model") +
    ggplot2::theme_minimal()
}

plot_difference_vs_chroma <- function(paired_data) {
  ggplot2::ggplot(paired_data, ggplot2::aes(x = .data$chroma, y = .data$D)) +
    ggplot2::geom_point(alpha = 0.35, color = "#4C78A8") +
    ggplot2::geom_smooth(method = "lm", se = TRUE, color = "#D55E00") +
    ggplot2::geom_hline(yintercept = 0, linetype = "dashed") +
    ggplot2::labs(x = "Chroma", y = "D = error gpt-4o-mini - error gpt-4o") +
    ggplot2::theme_minimal()
}

plot_lm_diagnostics <- function(model, title = "Model lineal") {
  old_par <- par(no.readonly = TRUE)
  on.exit(par(old_par))

  par(mfrow = c(2, 2))
  plot(model, which = 1, main = paste(title, "- residus vs prediccions"))
  plot(model, which = 2, main = paste(title, "- QQ plot"))
  hist(stats::rstandard(model), main = paste(title, "- residus estandarditzats"), xlab = "Residu")
  plot(stats::rstandard(model), type = "l", main = paste(title, "- residus en ordre"), ylab = "Residu")
}
