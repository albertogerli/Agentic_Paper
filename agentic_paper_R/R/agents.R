# R/agents.R

# Ensure R6, httr2, jsonlite, logger, digest are available
# if (!requireNamespace("R6", quietly = TRUE)) install.packages("R6")
# if (!requireNamespace("httr2", quietly = TRUE)) install.packages("httr2")
# if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
# if (!requireNamespace("logger", quietly = TRUE)) install.packages("logger") # From utils
# if (!requireNamespace("digest", quietly = TRUE)) install.packages("digest") # For CachingAsyncAgent

library(R6)
library(httr2)
library(jsonlite)
library(logger)
library(digest) # For CachingAsyncAgent

# Source utilities if not loaded elsewhere (e.g. in main.R)
# This makes the file runnable standalone for testing, but in a project,
# dependencies are usually managed at a higher level.
if (!exists("with_retry") || !is.function(with_retry)) {
  # Attempt to source if running in an environment where R/utils.R is relative
  if(file.exists("R/utils.R")) source("R/utils.R")
  else if(file.exists("../R/utils.R")) source("../R/utils.R") # If in tests/ for example
  else warning("R/utils.R not found or with_retry not defined. Agent functionality might be impaired.")
}


Agent <- R6Class("Agent",
  public = list(
    name = NULL,
    instructions = NULL,
    model = NULL,
    temperature = NULL,
    client_details = NULL, # Stores API key and other related info (renamed from client for clarity)
    config_instance = NULL, # To access global config

    initialize = function(name, instructions, model, temperature = 0.7, config_instance = NULL) {
      self$name <- name
      self$instructions <- instructions
      self$model <- model
      self$temperature <- temperature
      self$config_instance <- config_instance

      if (!is.null(config_instance) && !is.null(config_instance$api_key) && config_instance$api_key != "") {
        # get_openai_client from utils.R just returns a list with the api_key
        self$client_details <- get_openai_client(config_instance$api_key)
      } else {
        log_warn("Agent '{self$name}': OpenAI client details not initialized - API key missing or empty in config.")
        self$client_details <- NULL
      }
    },

    run = function(message_content) {
      if (is.null(self$client_details) || is.null(self$client_details$api_key) || self$client_details$api_key == "") {
        log_error("Agent '{self$name}': OpenAI client not initialized or API key missing.")
        stop("OpenAI client not initialized for agent ", self$name)
      }
      if (is.null(message_content) || !is.character(message_content) || trimws(message_content) == "") {
        log_error("Agent '{self$name}': Message content must be a non-empty character string.")
        stop("Message content must be a non-empty character string for agent '", self$name, "'")
      }

      task_name <- paste("Agent", self$name, "run")

      current_temp <- if (self$model %in% c("o1-preview", "o1-mini")) 1.0 else self$temperature

      # Define the API call expression for with_retry
      # Ensure config_instance is available in this scope for req_timeout
      cfg_timeout <- self$config_instance$agent_timeout %||% 300

      api_call_expr <- quote({
        req <- request("https://api.openai.com/v1/chat/completions") %>%
          req_auth_bearer_token(token = self$client_details$api_key) %>%
          req_headers("Content-Type" = "application/json") %>%
          req_body_json(list(
            model = self$model,
            messages = list(
              list(role = "system", content = self$instructions),
              list(role = "user", content = message_content)
            ),
            temperature = current_temp,
            max_tokens = 4000
          )) %>%
          req_timeout(cfg_timeout)

        resp <- req_perform(req)

        if (resp_status(resp) >= 300) {
            status_code <- resp_status(resp)
            # Try to parse error body, otherwise use raw
            error_body_raw <- resp_body_raw(resp)
            error_body_string <- tryCatch(rawToChar(error_body_raw), error = function(e) "Failed to decode error body")

            log_error("Agent '{self$name}' API error: Status {status_code}, Body: {error_body_string}")
            stop(paste0("OpenAI API Error for agent '", self$name, "': ", status_code, " - ", error_body_string))
        }

        response_data <- resp_body_json(resp)

        if (is.null(response_data$choices) || length(response_data$choices) == 0 ||
            !is.list(response_data$choices[[1]]) || is.null(response_data$choices[[1]]$message$content)) {
            resp_str <- tryCatch(jsonlite::toJSON(response_data, auto_unbox = TRUE, pretty = TRUE), error = function(e) "Invalid JSON response")
            log_error("Agent '{self$name}': Unexpected API response structure: {resp_str}")
            stop(paste0("Unexpected API response structure from OpenAI for agent '", self$name, "'"))
        }

        result_content <- response_data$choices[[1]]$message$content
        result_content
      })

      if (!exists("with_retry", mode = "function")) {
        stop("with_retry function not found. Ensure R/utils.R is sourced and available.")
      }

      # The environment for eval needs to see 'self' or its relevant parts like client_details, model, instructions, temperature
      # We pass `self` itself to the environment of `with_retry`
      retry_env <- new.env(parent = parent.frame())
      retry_env$self <- self # Make the agent instance available
      retry_env$message_content <- message_content # And message_content
      retry_env$current_temp <- current_temp
      retry_env$cfg_timeout <- cfg_timeout


      final_result <- with_retry(
        expr = eval(api_call_expr, envir = retry_env),
        max_attempts = 3,
        initial_wait = 4,
        max_wait = 60,
        task_name = task_name
      )

      log_info("Agent '{self$name}' completed successfully.")
      return(final_result)
    }
  )
)

AsyncAgent <- R6Class("AsyncAgent",
  inherit = Agent,
  public = list(
    initialize = function(name, instructions, model, temperature = 0.7, config_instance = NULL) {
      super$initialize(name, instructions, model, temperature, config_instance)
    },
    arun = function(message_content) {
      log_debug("AsyncAgent$arun called for '{self$name}'. Currently executes synchronously via super$run().")
      return(super$run(message_content))
    }
  )
)

CachingAsyncAgent <- R6Class("CachingAsyncAgent",
  inherit = AsyncAgent,
  private = list(
    .cache = NULL
  ),
  public = list(
    initialize = function(name, instructions, model, temperature = 0.7, config_instance = NULL) {
      super$initialize(name, instructions, model, temperature, config_instance)
      private$.cache <- new.env(hash = TRUE, parent = emptyenv())
      log_info("CachingAsyncAgent '{self$name}' initialized with new cache.")
    },

    arun = function(message_content) {
      if (!exists("hash_message", mode="function")) {
        stop("hash_message function not found. Ensure R/utils.R is sourced.")
      }
      key <- hash_message(message_content)

      if (exists(key, envir = private$.cache, inherits = FALSE)) {
        log_info("Cache hit for agent '{self$name}' with key '{key}'.")
        return(private$.cache[[key]])
      }

      log_info("Cache miss for agent '{self$name}' with key '{key}'. Calling super$arun.")
      result <- super$arun(message_content)

      private$.cache[[key]] <- result
      log_info("Result stored in cache for agent '{self$name}' with key '{key}'.")
      return(result)
    }
  )
)

# Helper for config default: x %||% y -> if x is NULL, use y, else use x
# Moved to utils.R to avoid redefinition if agents.R is sourced multiple times.
# Ensure it's available or define it in utils.R and source utils.R
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) {
    if (is.null(x)) y else x
  }
}
```
**Note:** I've made the `arun` in `AsyncAgent` call `super$run()` for now. The asynchronicity using `future` will be handled in the `ReviewOrchestrator` by wrapping calls to these agent methods. The `retry_env` was added to ensure `self` and other necessary variables are correctly scoped when `eval(api_call_expr)` is executed by `with_retry`. I also added `digest` to the list of libraries at the top.Okay, continuing with the translation of Python classes and functions to R.

Next up is `PaperInfo` (which will be a simple R6 class or even a named list function, but R6 is cleaner for consistency) and `FileManager`.

**5. `R/paper_info.R`** (for `PaperInfo`)

```R
# R/paper_info.R
library(R6)
library(jsonlite) # For to_json method

PaperInfo <- R6Class("PaperInfo",
  public = list(
    title = "Unknown title",
    authors = "Unknown authors",
    abstract = "Abstract not found",
    length = 0, # Character length
    sections = list(),
    file_path = NULL,

    initialize = function(title = "Unknown title", authors = "Unknown authors",
                          abstract = "Abstract not found", length = 0,
                          sections = list(), file_path = NULL) {
      self$title <- title
      self$authors <- authors
      self$abstract <- abstract
      self$length <- length
      self$sections <- sections
      self$file_path <- file_path
    },

    to_list = function() {
      list(
        title = self$title,
        authors = self$authors,
        abstract = self$abstract,
        length = self$length,
        sections = self$sections,
        file_path = self$file_path
      )
    },

    to_json = function(pretty = TRUE, auto_unbox = TRUE) {
      jsonlite::toJSON(self$to_list(), pretty = pretty, auto_unbox = auto_unbox)
    },

    print = function(...) {
      cat("PaperInfo:\n")
      cat("  Title: ", self$title, "\n")
      cat("  Authors: ", self$authors, "\n")
      cat("  Abstract: ", substr(self$abstract, 1, 100), "...\n")
      cat("  Length: ", self$length, " characters\n")
      cat("  Sections (first 5): ", paste(head(self$sections, 5), collapse = ", "), "\n")
      if (!is.null(self$file_path)) {
        cat("  File Path: ", self$file_path, "\n")
      }
      invisible(self)
    }
  )
)
```

**6. `R/file_manager.R`**

```R
# R/file_manager.R

# Ensure R6, jsonlite, pdftools, logger are available
# if (!requireNamespace("R6", quietly = TRUE)) install.packages("R6")
# if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")
# if (!requireNamespace("pdftools", quietly = TRUE)) install.packages("pdftools")
# if (!requireNamespace("logger", quietly = TRUE)) install.packages("logger")

library(R6)
library(jsonlite)
library(pdftools) # For PDF extraction
library(logger)

# Assuming utils.R (with ensure_dir_exists, safe_list_files, read_file_with_encodings) is sourced
if (!exists("ensure_dir_exists", mode="function")) {
  if(file.exists("R/utils.R")) source("R/utils.R")
  else if(file.exists("../R/utils.R")) source("../R/utils.R")
  else warning("R/utils.R not found. FileManager functionality might be impaired.")
}


FileManager <- R6Class("FileManager",
  public = list(
    output_dir = NULL,

    initialize = function(output_dir) {
      self$output_dir <- output_dir
      ensure_dir_exists(self$output_dir) # from utils.R
    },

    save_json = function(data_list, filename, pretty = TRUE, auto_unbox = TRUE) {
      filepath <- file.path(self$output_dir, filename)
      tryCatch({
        json_data <- jsonlite::toJSON(data_list, pretty = pretty, auto_unbox = auto_unbox)
        writeLines(json_data, filepath, useBytes = TRUE) # useBytes for UTF-8 consistency
        log_info("JSON saved: {filepath}")
        return(TRUE)
      }, error = function(e) {
        log_error("Error saving JSON {filepath}: {conditionMessage(e)}")
        return(FALSE)
      })
    },

    save_text = function(text_content, filename) {
      filepath <- file.path(self$output_dir, filename)
      tryCatch({
        # Ensure text_content is a single string
        if (is.list(text_content) || length(text_content) > 1) {
            text_content <- paste(unlist(text_content), collapse = "\n")
        }
        writeLines(as.character(text_content), filepath, useBytes = TRUE) # useBytes for UTF-8
        log_info("Text file saved: {filepath}")
        return(TRUE)
      }, error = function(e) {
        log_error("Error saving text file {filepath}: {conditionMessage(e)}")
        return(FALSE)
      })
    },

    extract_text_from_pdf = function(pdf_path) {
      if (!file.exists(pdf_path)) {
        log_error("PDF not found: {pdf_path}")
        return("") # Return empty string as per Python version
      }
      tryCatch({
        # pdftools::pdf_text returns a character vector, one element per page
        text_pages <- pdftools::pdf_text(pdf_path)
        # Concatenate pages, Python version adds "\n\n"
        full_text <- paste(text_pages, collapse = "\n\n")
        log_info("Text extracted successfully from PDF: {pdf_path}")
        return(full_text)
      }, error = function(e) {
        log_error("PDF extraction failed for {pdf_path}: {conditionMessage(e)}")
        return("")
      })
    },

    save_review = function(reviewer_name, review_content) {
      # Ensure the 'reviews' subdirectory exists
      reviews_subdir <- file.path(self$output_dir, "reviews")
      ensure_dir_exists(reviews_subdir)

      filename <- paste0("review_", gsub("\\s+", "_", trimws(reviewer_name)), ".txt")
      # Save within the 'reviews' subdirectory
      filepath <- file.path(reviews_subdir, filename)

      success <- self$save_text(review_content, filepath) # save_text prepends output_dir already
      # so we need to adjust the call or the save_text function.
      # Let's adjust the call here for simplicity for now.
      # We'll pass the relative path from output_dir to save_text.

      success_alt <- tryCatch({
        # Ensure text_content is a single string
        if (is.list(review_content) || length(review_content) > 1) {
            review_content <- paste(unlist(review_content), collapse = "\n")
        }
        writeLines(as.character(review_content), filepath, useBytes = TRUE)
        log_info("Review text file saved: {filepath}")
        TRUE
      }, error = function(e) {
        log_error("Error saving review text file {filepath}: {conditionMessage(e)}")
        FALSE
      })

      if (success_alt) {
        return(paste0("Review successfully saved in ", filename, " (within reviews subdir)"))
      } else {
        return(paste0("Error saving review for ", reviewer_name))
      }
    },

    get_reviews = function() {
      reviews <- list()
      reviews_dir <- file.path(self$output_dir, "reviews")

      if (!dir.exists(reviews_dir)) {
        log_warn("Reviews directory does not exist: {reviews_dir}")
        return(reviews)
      }

      review_files <- safe_list_files(reviews_dir, pattern = "^review_.*\\.txt$", full.names = TRUE) # from utils.R

      for (filepath in review_files) {
        tryCatch({
          # Extract reviewer name from filename: review_N_A_M_E.txt -> N A M E
          filename_sans_ext <- sub("\\.txt$", "", basename(filepath))
          reviewer_name_raw <- sub("^review_", "", filename_sans_ext)
          reviewer_name <- gsub("_", " ", reviewer_name_raw)

          # Read file content (assuming UTF-8, which save_text uses)
          # Using read_file_with_encodings for robustness, though save_text aims for UTF-8
          file_content <- read_file_with_encodings(filepath) # from utils.R
          if (!is.null(file_content)) {
            reviews[[reviewer_name]] <- file_content
          } else {
            log_warn("Could not read review file: {filepath}")
          }
        }, error = function(e) {
          log_error("Error reading review {filepath}: {conditionMessage(e)}")
        })
      }
      return(reviews)
    },

    read_paper = function(file_path) {
      # This uses the utility function from utils.R
      return(read_file_with_encodings(file_path))
    }
  )
)
```

Creating these files now.
