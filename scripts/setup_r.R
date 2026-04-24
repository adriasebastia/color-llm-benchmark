options(repos = c(CRAN = "https://cloud.r-project.org"))

packages <- readLines("requirements-r.txt")
packages <- packages[packages != "" & !startsWith(packages, "#")]

install.packages(packages)

if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

renv::init(bare = TRUE)
renv::install(packages)
renv::snapshot(prompt = FALSE)

IRkernel::installspec(
  name = "color-llm-benchmark-r",
  displayname = "R (color-llm-benchmark)",
  user = TRUE
)

message("R environment ready.")
