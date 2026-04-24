options(repos = c(CRAN = "https://cloud.r-project.org"))

user_library <- Sys.getenv("R_LIBS_USER")
if (!dir.exists(user_library)) {
  dir.create(user_library, recursive = TRUE, showWarnings = FALSE)
}
.libPaths(c(user_library, .libPaths()))

project_root <- normalizePath(getwd(), winslash = "/", mustWork = TRUE)
jupyter_bin <- file.path(project_root, ".venv", "Scripts")
if (dir.exists(jupyter_bin)) {
  Sys.setenv(PATH = paste(jupyter_bin, Sys.getenv("PATH"), sep = .Platform$path.sep))
}

packages <- readLines("requirements-r.txt")
packages <- packages[packages != "" & !startsWith(packages, "#")]
packages <- setdiff(packages, "renv")

install.packages(packages)

IRkernel::installspec(
  name = "color-llm-benchmark-r",
  displayname = "R (color-llm-benchmark)",
  user = TRUE
)

message("R environment ready.")
