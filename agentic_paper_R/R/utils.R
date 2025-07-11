# R/utils.R

# Ensure logger is installed
# if (!requireNamespace("logger", quietly = TRUE)) {
#   install.packages("logger")
# }

library(logger)

# Global logger instance (or configure as needed)
# Defaulting to console, file logging can be added as in Python
setup_logging <- function(log_level = "INFO", log_file = "logs/paper_review_system.log") {
  log_threshold(log_level) # Sets the minimum level to log

  # Clear existing appenders to avoid duplicate messages if re-run
  log_appender(appender_console, index = 1) # Keep console

  # Remove file appender if it exists by default or from previous calls
  # This is a bit tricky as logger doesn't have a simple remove_appender_by_name
  # For simplicity, we'll just add ours. If running setup_logging multiple times,
  # it might add multiple file appenders. A more robust solution would manage this.

  # File appender
  log_dir <- dirname(log_file)
  if (!dir.exists(log_dir)) {
    dir.create(log_dir, recursive = TRUE)
  }
  # Ensure we don't add duplicate file appenders if function is called multiple times
  # A more robust way would be to manage appender names if logger supported it easily.
  # For now, we'll assume setup_logging is called once at the beginning.
  if (length(log_appenders()) < 2) { # Crude check: if only console appender exists
    log_appender(appender_file(log_file), index = 2)
  }


  # Default is fine, or customize:
  # log_formatter(formatter_glue_line)
  # formatter_glue_line <- function(level, msg, namespace = "global", .logcall = sys.call(), .topcall = sys.call(-1), .topenv = parent.frame()) {
  #   paste0(
  #     format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " - ",
  #     level, " - ",
  #     "[", namespace, "] - ",
  #     msg, "\n"
  #   )
  # }
  log_formatter(formatter_glue) # Using the default glue formatter

  # Initial log message
  log_info("Logging system configured. Level: {log_level}, File: {log_file}")
}

# Helper for OpenAI client initialization (to be used by Agent class)
# This will use httr2 for making requests.
# The "client" will be a list containing the API key and other relevant info.
get_openai_client <- function(api_key) {
  if (is.null(api_key) || api_key == "") {
    log_warn("OpenAI API key is not set. Client will not be functional.")
    return(NULL)
  }
  # For httr2, the "client" is more conceptual; we'd store the key and use it in requests.
  return(list(api_key = api_key))
}

# Retry mechanism
# For more complex needs, consider the `retry` package.
with_retry <- function(expr, max_attempts = 3, initial_wait = 4, max_wait = 60, task_name = "Unnamed task") {
  attempt <- 1
  wait_time <- initial_wait

  # Capture the expression and its environment
  expr_captured <- substitute(expr)
  env <- parent.frame()

  while (attempt <= max_attempts) {
    result <- tryCatch({
      eval(expr_captured, envir = env) # Evaluate the expression in the calling environment
    }, error = function(e) {
      log_warn("Attempt {attempt} for '{task_name}' failed: {conditionMessage(e)}")
      if (attempt == max_attempts) {
        log_error("Max attempts reached for '{task_name}'. Failing.")
        stop(e) # Re-throw the error
      }
      return(structure(list(error = e, task_name = task_name, attempt = attempt), class = "retry_error_signal"))
    })

    if (!inherits(result, "retry_error_signal")) {
      return(result)
    }

    log_info("Waiting {wait_time}s before retrying '{task_name}' (next attempt: {attempt + 1})")
    Sys.sleep(wait_time)
    attempt <- attempt + 1
    wait_time <- min(wait_time * 2, max_wait) # Exponential backoff
  }
}

# Function to hash a message for caching (simple approach)
hash_message <- function(message_content) {
  if (!requireNamespace("digest", quietly = TRUE)) {
    # install.packages("digest") # Or declare as dependency
    stop("Package 'digest' needed for hashing. Please install it.")
  }
  # Ensure message is a single character string for consistent hashing
  if (is.list(message_content) || length(message_content) > 1) {
    message_content <- paste(unlist(message_content), collapse = " ")
  } else if (!is.character(message_content)) {
    message_content <- as.character(message_content)
  }
  return(digest::digest(message_content, algo = "md5"))
}

# Safe version of list.files that returns empty list if dir doesn't exist
safe_list_files <- function(path, pattern = NULL, all.files = FALSE,
                            full.names = FALSE, recursive = FALSE,
                            ignore.case = FALSE, include.dirs = FALSE,
                            no.. = FALSE) {
  if (!dir.exists(path)) {
    return(character(0))
  }
  list.files(path = path, pattern = pattern, all.files = all.files,
             full.names = full.names, recursive = recursive,
             ignore.case = ignore.case, include.dirs = include.dirs,
             no.. = no..)
}

# Helper to ensure a directory exists
ensure_dir_exists <- function(dir_path) {
  if (!dir.exists(dir_path)) {
    dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
    log_info("Created directory: {dir_path}")
  }
}

# Helper to read file with multiple encodings
read_file_with_encodings <- function(file_path, encodings = c('UTF-8', 'latin1', 'CP1252', 'ISO-8859-1')) {
  if (!file.exists(file_path)) {
    log_error("File not found: {file_path}")
    return(NULL)
  }

  content <- NULL
  for (enc in encodings) {
    tryCatch({
      conn <- file(file_path, open = "r", encoding = enc)
      content <- paste(readLines(conn, warn = FALSE), collapse = "\n")
      close(conn)
      log_info("File {file_path} read successfully with {enc} encoding")
      return(content)
    }, error = function(e) {
      # Try next encoding
      log_debug("Failed to read {file_path} with {enc}: {conditionMessage(e)}")
      if (!is.null(conn) && isOpen(conn)) {
        close(conn)
      }
    })
  }

  log_error("Could not read file {file_path} with any of the attempted encodings.")
  return(NULL)
}
