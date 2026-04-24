# Funcions petites per la recollida de dades.
# La idea es que el notebook no acabi ple de codi repetit.

write_log <- function(message, log_file = file.path("logs", "pipeline.log")) {
  log_dir <- dirname(log_file)

  if (!dir.exists(log_dir)) {
    dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
  }

  line <- paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), message)
  write(line, file = log_file, append = TRUE)
  invisible(line)
}

save_csv <- function(data, path) {
  if (!inherits(data, "data.frame")) {
    stop("save_csv espera un data.frame o tibble.")
  }

  output_dir <- dirname(path)
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  readr::write_csv(data, path)
  invisible(path)
}

download_images <- function(data, url_col = "image_url", output_dir = "images") {
  if (!inherits(data, "data.frame")) {
    stop("download_images espera un data.frame o tibble.")
  }

  if (!url_col %in% names(data)) {
    stop(paste("No existe la columna", url_col))
  }

  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  }

  warning("download_images es una estructura base: falta implementar la descarga real.")
  invisible(file.path(output_dir))
}
