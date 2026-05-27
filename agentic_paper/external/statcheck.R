#!/usr/bin/env Rscript
# statcheck.R — small wrapper used by agentic_paper.external.statcheck
#
# Reads paper text from stdin, runs the `statcheck` R package on it, and
# emits a single JSON object on stdout describing what it found. The Python
# side calls this via subprocess and never inspects R's stderr — so we always
# exit cleanly and put the diagnostic into the JSON ``reason`` field.

suppressMessages(suppressWarnings({
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    cat('{"available": false, "reason": "R package jsonlite not installed (install.packages(\\\"jsonlite\\\"))"}')
    quit(status = 0)
  }
  if (!requireNamespace("statcheck", quietly = TRUE)) {
    cat(jsonlite::toJSON(list(
      available = FALSE,
      reason = "R package statcheck not installed (install.packages(\"statcheck\"))"
    ), auto_unbox = TRUE))
    quit(status = 0)
  }
  library(jsonlite)
  library(statcheck)
}))

# Read all of stdin into one string.
text <- paste(readLines(file("stdin"), warn = FALSE), collapse = "\n")
if (nchar(text) < 10) {
  cat(jsonlite::toJSON(list(
    available = TRUE,
    n_stats = 0,
    n_errors = 0,
    n_decision_errors = 0,
    rows = list()
  ), auto_unbox = TRUE, na = "null"))
  quit(status = 0)
}

result <- tryCatch({
  statcheck::statcheck(text, messages = FALSE)
}, error = function(e) {
  message("statcheck error: ", conditionMessage(e))
  return(NULL)
}, warning = function(w) {
  invokeRestart("muffleWarning")
})

if (is.null(result) || nrow(result) == 0) {
  cat(jsonlite::toJSON(list(
    available = TRUE,
    n_stats = 0,
    n_errors = 0,
    n_decision_errors = 0,
    rows = list()
  ), auto_unbox = TRUE, na = "null"))
  quit(status = 0)
}

# statcheck's column names vary across versions:
#   v2.x (current CRAN): source, test_type, df1, df2, test_comp, test_value,
#                        p_comp, reported_p, computed_p, raw, error, decision_error
#   v1.x:                Source, Statistic, df1, df2, Value, Reported.Comparison,
#                        Reported.P.Value, Computed, Raw, Error, DecisionError
get_col <- function(df, names, default = NA) {
  for (n in names) {
    if (n %in% colnames(df)) return(df[[n]])
  }
  return(rep(default, nrow(df)))
}

statistic_col       <- get_col(result, c("test_type", "Statistic"))
df1_col             <- get_col(result, c("df1"))
df2_col             <- get_col(result, c("df2"))
value_col           <- get_col(result, c("test_value", "Value"))
reported_comp_col   <- get_col(result, c("p_comp", "Reported.Comparison"), "=")
reported_p_col      <- get_col(result, c("reported_p", "Reported.P.Value"))
computed_p_col      <- get_col(result, c("computed_p", "Computed"))
error_col           <- get_col(result, c("error", "Error"), FALSE)
decision_error_col  <- get_col(result, c("decision_error", "DecisionError"), FALSE)
raw_col             <- get_col(result, c("raw", "Raw"), "")

# Helper: emit NA (jsonlite renders as JSON null with na="null") instead of
# NULL (which jsonlite renders as an empty list "{}" — junk for the Python side).
maybe_num <- function(x) if (length(x) == 0 || is.na(x)) NA_real_ else as.numeric(x)
maybe_str <- function(x) if (length(x) == 0 || is.na(x)) NA_character_ else as.character(x)

rows <- lapply(seq_len(nrow(result)), function(i) {
  list(
    statistic = maybe_str(statistic_col[i]),
    df1 = maybe_num(df1_col[i]),
    df2 = maybe_num(df2_col[i]),
    value = maybe_num(value_col[i]),
    reported_comparison = maybe_str(reported_comp_col[i]),
    reported_p_value = maybe_num(reported_p_col[i]),
    computed_p_value = maybe_num(computed_p_col[i]),
    error = isTRUE(error_col[i]),
    decision_error = isTRUE(decision_error_col[i]),
    raw = maybe_str(raw_col[i])
  )
})

cat(jsonlite::toJSON(list(
  available = TRUE,
  n_stats = nrow(result),
  n_errors = as.integer(sum(error_col == TRUE, na.rm = TRUE)),
  n_decision_errors = as.integer(sum(decision_error_col == TRUE, na.rm = TRUE)),
  rows = rows
), auto_unbox = TRUE, na = "null"))
