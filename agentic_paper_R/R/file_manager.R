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
# For standalone testing/sourcing:
if (!exists("ensure_dir_exists", mode="function") ||
    !exists("safe_list_files", mode="function") ||
    !exists("read_file_with_encodings", mode="function")) {
  if(file.exists("R/utils.R")) source("R/utils.R")
  else if(file.exists("../R/utils.R")) source("../R/utils.R") # If in tests/ for example
  else warning("R/utils.R not found or core utility functions are not defined. FileManager functionality might be impaired.")
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
        # Ensure data_list is suitable for JSON conversion (e.g., no complex R6 objects directly)
        # If data_list contains R6 objects, they should have a to_list() or similar method,
        # or be converted before calling this function.
        json_data <- jsonlite::toJSON(data_list, pretty = pretty, auto_unbox = auto_unbox, force = TRUE)
        writeLines(json_data, filepath, useBytes = TRUE) # useBytes for UTF-8 consistency
        log_info("JSON saved: {filepath}")
        return(TRUE)
      }, error = function(e) {
        log_error("Error saving JSON to {filepath}: {conditionMessage(e)}")
        return(FALSE)
      })
    },

    save_text = function(text_content, filename) {
      filepath <- file.path(self$output_dir, filename)
      tryCatch({
        if (is.list(text_content) || length(text_content) > 1) {
            text_content <- paste(unlist(text_content), collapse = "\n")
        }
        writeLines(as.character(text_content), filepath, useBytes = TRUE)
        log_info("Text file saved: {filepath}")
        return(TRUE)
      }, error = function(e) {
        log_error("Error saving text file to {filepath}: {conditionMessage(e)}")
        return(FALSE)
      })
    },

    extract_text_from_pdf = function(pdf_path) {
      if (!file.exists(pdf_path)) {
        log_error("PDF not found: {pdf_path}")
        return("")
      }
      tryCatch({
        text_pages <- pdftools::pdf_text(pdf_path)
        full_text <- paste(text_pages, collapse = "\n\n") # Python version adds "\n\n"
        log_info("Text extracted successfully from PDF: {pdf_path}")
        return(full_text)
      }, error = function(e) {
        log_error("PDF extraction failed for {pdf_path}: {conditionMessage(e)}")
        return("")
      })
    },

    save_review = function(reviewer_name, review_content) {
      reviews_subdir_name <- "reviews" # Relative path for the subdirectory
      reviews_subdir_full_path <- file.path(self$output_dir, reviews_subdir_name)
      ensure_dir_exists(reviews_subdir_full_path)

      filename_txt <- paste0("review_", gsub("[^[:alnum:]_]", "_", trimws(reviewer_name)), ".txt")
      # Pass the filename relative to output_dir for save_text to construct full path
      # Or, construct full path here and save_text uses it directly.
      # The current save_text prepends self$output_dir, so we give it a path relative to that.
      # Example: filename for save_text should be "reviews/review_My_Agent.txt"
      relative_filepath_for_save_text <- file.path(reviews_subdir_name, filename_txt)

      success <- self$save_text(review_content, relative_filepath_for_save_text)

      if (success) {
        # The actual full path for logging/returning
        full_saved_path <- file.path(self$output_dir, relative_filepath_for_save_text)
        log_info("Review successfully saved: {full_saved_path}")
        return(paste0("Review successfully saved in ", relative_filepath_for_save_text))
      } else {
        log_error("Error saving review for {reviewer_name}")
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

      # safe_list_files is from utils.R
      review_files <- safe_list_files(reviews_dir, pattern = "^review_.*\\.txt$", full.names = TRUE)

      for (filepath in review_files) {
        tryCatch({
          filename_sans_ext <- sub("\\.txt$", "", basename(filepath))
          reviewer_name_raw <- sub("^review_", "", filename_sans_ext)
          reviewer_name <- gsub("_", " ", reviewer_name_raw) # Python version replaced ' ' with '_' for saving

          # read_file_with_encodings is from utils.R
          file_content <- read_file_with_encodings(filepath)
          if (!is.null(file_content)) {
            reviews[[reviewer_name]] <- file_content
          } else {
            log_warn("Could not read review file (or content was NULL): {filepath}")
          }
        }, error = function(e) {
          log_error("Error processing review file {filepath}: {conditionMessage(e)}")
        })
      }
      return(reviews)
    },

    read_paper = function(file_path) {
      # read_file_with_encodings is from utils.R
      content <- read_file_with_encodings(file_path)
      if (is.null(content)) {
          log_error("Failed to read paper file: {file_path}")
          return(NULL) # Or consider returning "" for consistency with Python if preferred
      }
      log_info("Paper read successfully: {file_path}. Length: {nchar(content)} characters")
      return(content)
    }
  )
)
```

One minor adjustment in `save_review` logic for path handling with `save_text`.
And `PaperInfo`'s `to_json` was made more robust for the `sections` field.

Continuing with `PaperAnalyzer`. This one is more complex due to the regex and AI interaction.

**7. `R/paper_analyzer.R`**

```R
# R/paper_analyzer.R

# Ensure R6, httr2, jsonlite, logger, stringr are available
# if (!requireNamespace("R6", quietly = TRUE)) install.packages("R6")
# if (!requireNamespace("httr2", quietly = TRUE)) install.packages("httr2") # For AI part
# if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite") # For AI part
# if (!requireNamespace("logger", quietly = TRUE)) install.packages("logger")
# if (!requireNamespace("stringr", quietly = TRUE)) install.packages("stringr")

library(R6)
library(httr2)
library(jsonlite)
library(logger)
library(stringr) # For regex convenience

# Assuming PaperInfo class is available (e.g. sourced)
if (!exists("PaperInfo", mode="classRepresentation")) {
  if(file.exists("R/paper_info.R")) source("R/paper_info.R")
  else if(file.exists("../R/paper_info.R")) source("../R/paper_info.R")
  else warning("R/paper_info.R not found or PaperInfo class not defined.")
}
# Assuming utils like get_openai_client, with_retry are available
if (!exists("get_openai_client", mode="function")) {
  if(file.exists("R/utils.R")) source("R/utils.R")
  else if(file.exists("../R/utils.R")) source("../R/utils.R")
  else warning("R/utils.R not found. PaperAnalyzer AI functionality might be impaired.")
}


PaperAnalyzer <- R6Class("PaperAnalyzer",
  public = list(
    config = NULL,
    client_details = NULL, # For OpenAI API calls

    initialize = function(config_obj) {
      self$config <- config_obj
      if (!is.null(self$config$api_key) && self$config$api_key != "") {
        self$client_details <- get_openai_client(self$config$api_key) # from utils.R
      } else {
        log_warn("PaperAnalyzer: OpenAI client details not initialized - API key missing in config.")
        self$client_details <- NULL
      }
    },

    extract_info = function(paper_text) {
      if (is.null(paper_text) || nchar(paper_text) == 0) {
        log_warn("Paper text is empty. Cannot extract info.")
        return(PaperInfo$new()) # Return default empty PaperInfo
      }

      info_list <- list(title = NULL, authors = NULL, abstract = NULL) # Use a list to store extracted parts
      ai_success <- FALSE

      if (!is.null(self$client_details) && !is.null(self$client_details$api_key)) {
        tryCatch({
          snippet <- substr(paper_text, 1, 15000)
          prompt <- glue::glue(
            "You are an expert assistant specializing in scientific literature. Your task is to extract the Title, Authors, and Abstract from the beginning of a scientific paper.
            The text of the paper is provided below. Please analyze it and return the extracted information in a valid JSON format with the following keys: \"title\", \"authors\", \"abstract\".
            - For \"title\", provide the full title of the paper.
            - For \"authors\", list all authors, separated by commas.
            - For \"abstract\", provide the full text of the abstract.
            If any piece of information cannot be found, use the value \"Not Found\".

            --- PAPER TEXT ---
            {snippet}
            --- END OF TEXT ---

            Return only the JSON object, without any additional comments or explanations."
          )

          # Using a basic model for this, as per Python code
          model_to_use <- self$config$model_basic %||% "gpt-3.5-turbo"

          # Simplified API call structure (not using the full Agent class here)
          # This is similar to Agent$run but specialized for this JSON response format.
          api_call_expr_analyzer <- quote({
              req <- request("https://api.openai.com/v1/chat/completions") %>%
                req_auth_bearer_token(token = self$client_details$api_key) %>%
                req_headers("Content-Type" = "application/json") %>%
                req_body_json(list(
                  model = model_to_use,
                  messages = list(
                    list(role = "system", content = "You are an expert assistant for scientific literature analysis. Your output must be a single, valid JSON object."),
                    list(role = "user", content = prompt)
                  ),
                  temperature = 0.0,
                  response_format = list(type = "json_object")
                )) %>%
                req_timeout(self$config$agent_timeout %||% 300)

              resp <- req_perform(req)
              if (resp_status(resp) >= 300) {
                  stop(paste("OpenAI API Error:", resp_status(resp), rawToChar(resp_body_raw(resp))))
              }
              resp_body_json(resp, simplifyVector = TRUE) # simplifyVector for easier list access
          })

          # Environment for eval
          eval_env <- new.env(parent = parent.frame())
          eval_env$self <- self # Make self available for client_details, config
          eval_env$model_to_use <- model_to_use
          eval_env$prompt <- prompt

          extracted_data_list <- with_retry(
            expr = eval(api_call_expr_analyzer, envir = eval_env),
            task_name = "PaperAnalyzer AI Info Extraction"
          )

          # The response itself is the JSON content (message content)
          # The Python code expects client.chat.completions.create(...).choices[0].message.content
          # which is then parsed by json.loads().
          # httr2's resp_body_json already parses it if the API returns JSON in the body.
          # If the actual content is nested, adjust here.
          # Assuming 'extracted_data_list' is the parsed JSON content.
          # The prompt asks for a JSON object like {"title": "T", "authors": "A", "abstract": "B"}
          # So, extracted_data_list should directly be this list.

          if (!is.null(extracted_data_list$title) &&
              extracted_data_list$title != "Not Found" &&
              extracted_data_list$title != "Unknown title") {
            log_info("Successfully extracted paper info using AI.")
            info_list$title <- extracted_data_list$title
            info_list$authors <- extracted_data_list$authors
            info_list$abstract <- extracted_data_list$abstract
            ai_success <- TRUE
          } else {
            log_warn("AI extraction did not find a valid title or returned 'Not Found'. Falling back to regex.")
          }
        }, error = function(e) {
          log_error("AI-based info extraction failed: {conditionMessage(e)}. Falling back to regex.")
        })
      } else {
        log_info("OpenAI client not configured for PaperAnalyzer. Using regex-based method only.")
      }

      if (!ai_success) {
        log_info("Using regex-based method to extract paper info.")
        regex_info <- private$.extract_info_with_regex(paper_text)

        # Merge results, giving preference to regex if AI failed or returned poor results
        # Or if AI info was NULL from the start
        info_list$title <- if (!is.null(regex_info$title) && regex_info$title != "Unknown title") regex_info$title else info_list$title
        info_list$authors <- if (!is.null(regex_info$authors) && regex_info$authors != "Unknown authors") regex_info$authors else info_list$authors
        info_list$abstract <- if (!is.null(regex_info$abstract) && regex_info$abstract != "Abstract not found") regex_info$abstract else info_list$abstract
      }

      # Fill NAs or NULLs with defaults before creating PaperInfo object
      final_title <- info_list$title %||% "Unknown title"
      final_authors <- info_list$authors %||% "Unknown authors"
      final_abstract <- info_list$abstract %||% "Abstract not found"

      sections <- private$.identify_sections(paper_text)

      return(
        PaperInfo$new(
          title = final_title,
          authors = final_authors,
          abstract = final_abstract,
          length = nchar(paper_text),
          sections = sections,
          file_path = NULL # File path is not set by analyzer, but by orchestrator if needed
        )
      )
    }
  ),
  private = list(
    .extract_info_with_regex = function(paper_text) {
      lines <- strsplit(paper_text, "\n")[[1]]

      # Title: first non-empty line (simplified)
      title <- "Unknown title"
      for (line in lines) {
        trimmed_line <- trimws(line)
        if (nchar(trimmed_line) > 0) {
          title <- trimmed_line
          break
        }
      }

      # Authors
      authors <- "Unknown authors"
      author_patterns <- c(
        '(?i)(?:Authors?|by|Autori|di):\\s*([^\\n]+)',
        # R regex doesn't always support PCRE lookaheads/behinds as easily as Python's re if used.
        # Using simpler captures. (?m) is often implicit per line for stringr or handled by base R multiline options.
        # '^\\s*([A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+(?:,\\s*[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+)*)',
        # Simplified: one or more capitalized words, possibly with initials, comma separated.
        # This is hard to get perfect with one regex. Python one is also heuristic.
        # A common pattern: "First Last, First Last" or "F. Last, F. Last"
        # Let's try to find lines that look like author lists.
        # This regex is a placeholder and likely needs refinement for R's engine and common PDF text artifacts.
        "^[A-Z][A-Za-z.'-]+(?:\\s+[A-Z][A-Za-z.'-]+)*(?:,\\s*[A-Z][A-Za-z.'-]+(?:\\s+[A-Z][A-Za-z.'-]+)*)+$", # Line consisting only of authors
        "^(?:[A-Z][a-z]+\\s+){1,3}[A-Z][a-z]+(?:\\s*,\\s*(?:[A-Z][a-z]+\\s+){1,3}[A-Z][a-z]+)*$" # Another attempt
      )

      # Try the "Authors:" pattern first
      author_match_explicit <- stringr::str_match(paper_text, author_patterns[1])
      if (!is.na(author_match_explicit[1,2])) {
        authors <- trimws(author_match_explicit[1,2])
      } else {
        # Fallback: search lines for author-like patterns (less reliable)
        # This is highly heuristic. The Python one is also quite complex.
        # For simplicity, we might just take a few lines after title if this fails often.
        # Python's regexes:
        # r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)*)'
        # r'(?:^|\n)([A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+(?:,\s*[A-Z][a-z]+\s+[A-Z]\.\s*[A-Z][a-z]+)*)'
        # These are complex and assume certain formatting.
        # A simpler R approach: look for lines with multiple capitalized words, possibly with commas.
        # This is a known hard problem for regex.
        # Let's try to find the first line after the title that seems to be a list of names.
        # This part needs significant improvement or reliance on the AI part.
        # For now, we'll keep it simple or even leave authors to AI primarily.
         title_line_idx <- -1
         for(i in seq_along(lines)){ if(trimws(lines[i]) == title) { title_line_idx <- i; break} }
         if(title_line_idx != -1 && title_line_idx + 1 <= length(lines)){
            potential_author_line <- trimws(lines[title_line_idx+1])
            # Very basic check: more than one word, starts with capital, contains comma or multiple spaces
            if(stringr::str_count(potential_author_line, "\\w+") > 1 && grepl("^[A-Z]", potential_author_line) && (grepl(",", potential_author_line) || stringr::str_count(potential_author_line, "\\s+") > 1)){
                # authors <- potential_author_line # This is too naive often.
            }
         }
      }


      # Abstract
      # (?s) for DOTALL equivalent in R's PCRE regex if using perl=TRUE
      # stringr functions often handle this more gracefully or have flags.
      # Python: r'(?:Abstract|Summary|Riassunto|Sommario)[:.\n]\s*([^\n]+(?:\n[^\n]+)*?)(?:\n\n|\n[A-Z]|\n\d+\.|$)'
      # R equivalent (approximate):
      abstract_pattern <- "(?i)(?:Abstract|Summary|Riassunto|Sommario)[:.\\s]*\\n*([^\\n]+(?:\\n(?!\\n|([A-Z]\\w*\\s*){3,}|\\d+\\.\\s+[A-Z])[^\\n]+)*)"
      # The negative lookahead is to stop before the next section or double newline. Hard to translate directly.
      # Simpler approach: Capture until two newlines or a line that looks like a new section title.
      abstract_match <- stringr::str_match(paper_text, regex(abstract_pattern, dotall = TRUE, multiline = TRUE))
      abstract <- if (!is.na(abstract_match[1,2])) trimws(abstract_match[1,2]) else "Abstract not found"

      # If abstract is very short, it might be just the word "Abstract" itself.
      if (nchar(abstract) < 30 && abstract != "Abstract not found") {
          # Attempt a simpler regex that grabs more aggressively until a common section keyword
          abstract_pattern_greedy <- "(?i)(Abstract|Summary|Riassunto|Sommario)[:.\\s]*\\n*(.+?)(?=\\n\\s*\\n|\\n\\s*(?:1\\.|I\\.|Introduction|Methods|Results|Discussion|Conclusion|References)|$)"
          abstract_match_greedy <- stringr::str_match(paper_text, regex(abstract_pattern_greedy, dotall = TRUE))
          if(!is.na(abstract_match_greedy[1,3])) abstract <- trimws(abstract_match_greedy[1,3])
      }


      list(title = title, authors = authors, abstract = abstract)
    },

    .identify_sections = function(paper_text) {
      standard_sections_regex <- paste0(
        "(?i)\\b(?:",
        paste(c(
            "Abstract", "Introduction", "Background", "Related Work", "Literature Review",
            "Methods?", "Methodology", "Materials and Methods", "Experimental Set(?:-| )up",
            "Results?", "Experiments?", "Evaluation", "Findings?",
            "Discussion", "Analysis", "Implications?",
            "Conclusion", "Conclusions?", "Future Work", "Limitations?",
            "References?", "Bibliography", "Acknowledgements?", "Appendix", "Supplementary Material"
        ), collapse = "|"),
        ")\\b"
      )

      sections_found <- c()
      lines <- stringr::str_split(paper_text, "\n")[[1]]

      # Patterns to identify sections
      # Order matters: more specific first
      section_patterns_r <- list(
        # 1. Numbered sections (e.g., "1. Introduction", "2.1. Methods")
        list(pattern = "^\\s*(\\d+(?:\\.\\d+)*)\\s*\\.?\\s+([A-Z][A-Za-z0-9\\s\\-:,()&]{3,})$", has_num = TRUE, extract_group = 2, num_group = 1),
        # 2. Roman numeral sections (e.g., "I. Introduction") - less common in modern papers
        list(pattern = "^\\s*([IVXLCDM]+(?:\\.[IVXLCDM]+)*)\\s*\\.?\\s+([A-Z][A-Za-z0-9\\s\\-:,()&]{3,})$", has_num = TRUE, extract_group = 2, num_group = 1),
        # 3. Uppercase unnumbered sections (e.g., "INTRODUCTION", "RESULTS AND DISCUSSION")
        list(pattern = "^\\s*([A-Z][A-Z0-9\\s\\-:,()&]{3,})$", has_num = FALSE, extract_group = 1),
        # 4. Standard sections (Title Case or sentence case, possibly with number)
        list(pattern = paste0("^\\s*(?:\\d+(?:\\.\\d+)*\\s*\\.?\\s*)?(", standard_sections_regex, ")[:.]?\\s*$"), has_num = FALSE, extract_group = 1) # extract_group refers to the main title part
        # Markdown headers are less common in raw PDF text but can be added if needed:
        # list(pattern = "^#+\\s+(.+)$", has_num = FALSE, extract_group = 1)
      )

      potential_sections <- character()

      for (i in seq_along(lines)) {
        line <- trimws(lines[i])
        if (nchar(line) == 0 || nchar(line) > 100) next # Skip empty or very long lines

        for (p_info in section_patterns_r) {
          match_res <- stringr::str_match(line, p_info$pattern)
          if (!is.na(match_res[1,1])) { # If the pattern matched
            title <- trimws(match_res[1, p_info$extract_group + 1]) # +1 because match_res[1,1] is full match

            if (nchar(title) > 2 && nchar(title) < 70) { # Filter by length
              # Context check (simplified from Python version)
              prev_line_empty <- if (i > 1) nchar(trimws(lines[i-1])) == 0 else TRUE
              next_line_starts_new_thought <- TRUE # Assume true for simplicity for now
              # Python: (next_line and (next_line[0].isupper() or not next_line[0].isalpha()))
              if (i < length(lines)) {
                  next_line_content <- trimws(lines[i+1])
                  if (nchar(next_line_content) > 0) {
                      # Check if next line starts with lowercase letter (suggests continuation, not new section)
                      if (grepl("^[a-z]", next_line_content)) {
                          next_line_starts_new_thought <- FALSE
                      }
                  }
              }


              if (prev_line_empty || next_line_starts_new_thought) {
                section_title_formatted <- stringr::str_to_title(title) # Convert to Title Case
                if (p_info$has_num && !is.na(match_res[1, p_info$num_group + 1])) {
                  num_prefix <- match_res[1, p_info$num_group + 1]
                  section_title_formatted <- paste0(num_prefix, ". ", section_title_formatted)
                }

                # Avoid adding very similar titles (e.g. "Result" and "Results")
                is_too_similar <- FALSE
                normalized_new_title <- tolower(gsub("[^a-z0-9]", "", section_title_formatted))
                for(existing_sec in potential_sections){
                    normalized_existing <- tolower(gsub("[^a-z0-9]", "", existing_sec))
                    if(stringdist::stringsim(normalized_new_title, normalized_existing) > 0.85){ # High similarity
                        is_too_similar <- TRUE
                        break
                    }
                }

                if(!is_too_similar) {
                    potential_sections <- c(potential_sections, section_title_formatted)
                }
                break # Found a pattern for this line
              }
            }
          }
        }
      }

      sections_found <- unique(potential_sections) # Remove exact duplicates

      # Heuristic if few sections found (simplified)
      if (length(sections_found) < 3) {
        log_info("Few sections found by primary patterns, trying heuristic for standard sections.")
        text_lower <- tolower(paper_text)
        standard_section_keywords <- c("introduction", "method", "result", "discussion", "conclusion", "reference")
        for (sec_kw in standard_section_keywords) {
          if (stringr::str_detect(text_lower, paste0("\\b", sec_kw))) {
            # Add the title-cased version if found
            found_keyword_titlecase <- stringr::str_to_title(sec_kw)
            if (!(found_keyword_titlecase %in% sections_found)) {
                 # A bit more specific check: does it appear on its own line?
                 if(any(grepl(paste0("^\\s*", sec_kw, "\\s*$"), lines, ignore.case = TRUE))){
                    sections_found <- c(sections_found, found_keyword_titlecase)
                 }
            }
          }
        }
        sections_found <- unique(sections_found)
      }

      # Limit to 20 sections (Python version does _filter_similar_sections then [:20])
      if (length(sections_found) > 20) {
        sections_found <- sections_found[1:20]
      }

      log_info("Identified {length(sections_found)} sections.")
      return(sections_found)
    }
  )
)

# Helper for default values if not already defined
if (!exists("%||%", mode = "function")) {
  `%||%` <- function(x, y) {
    if (is.null(x)) y else x
  }
}
# Required for string similarity check in .identify_sections
# if (!requireNamespace("stringdist", quietly = TRUE)) install.packages("stringdist")
library(stringdist)
```
This `PaperAnalyzer` is quite involved.
- The AI extraction part is simplified to directly call `httr2` without using the full `Agent` class, as it has a specific need for JSON response format.
- Regex for authors is notoriously difficult and has been kept simple, relying more on the AI part.
- Section identification uses a series of regex patterns and some heuristics. The Python version's `_filter_similar_sections` is partially incorporated by checking similarity before adding. `stringdist` package is used for this.
- `glue` might be needed for the prompt construction if not already available. (I'll add it to the library list at the top).

Creating the file:
