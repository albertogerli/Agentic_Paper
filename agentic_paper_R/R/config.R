# R/config.R

# Ensure yaml and R6 are installed
# if (!requireNamespace("yaml", quietly = TRUE)) install.packages("yaml")
# if (!requireNamespace("R6", quietly = TRUE)) install.packages("R6")

library(yaml)
library(R6)
library(logger) # Assuming utils.R and its setup_logging is sourced/available

Config <- R6Class("Config",
  public = list(
    api_key = NULL,
    model_powerful = "gpt-4o", # Updated default, was "o3"
    model_standard = "gpt-4-turbo", # Updated default, was "gpt-4.1"
    model_basic = "gpt-3.5-turbo", # Updated default, was "gpt-4.1-mini"
    output_dir = "output_revisione_paper", # Default, can be overridden
    max_parallel_agents = 3,
    agent_timeout = 300, # seconds
    temperature_methodology = 1.0,
    temperature_results = 1.0,
    temperature_literature = 1.0,
    temperature_structure = 1.0,
    temperature_impact = 1.0,
    temperature_contradiction = 1.0,
    temperature_ethics = 1.0,
    temperature_coordinator = 1.0,
    temperature_editor = 1.0,
    temperature_ai_origin = 1.0,
    temperature_hallucination = 1.0,

    initialize = function(config_path = NULL, output_dir_override = NULL) {
      # Attempt to load API key from environment variable first
      self$api_key <- Sys.getenv("OPENAI_API_KEY", unset = "")

      # Load from YAML if path is provided
      if (!is.null(config_path) && file.exists(config_path)) {
        tryCatch({
          config_data <- yaml::read_yaml(config_path)

          # Override fields if they exist in the YAML
          for (name in names(config_data)) {
            if (name %in% names(self)) {
              # Ensure that if api_key is in YAML, it overrides env var, even if env var had a value
              if (name == "api_key" && !is.null(config_data[[name]]) && config_data[[name]] != "") {
                 self$api_key <- config_data[[name]]
              } else if (name != "api_key") {
                 self[[name]] <- config_data[[name]]
              }
            }
          }
          log_info("Configuration loaded from {config_path}")
        }, error = function(e) {
          log_error("Error loading config from {config_path}: {conditionMessage(e)}. Using defaults and environment variables.")
        })
      } else {
        if (!is.null(config_path)) {
            log_warn("Config file {config_path} not found. Using defaults and environment variables.")
        } else {
            log_info("No config file path provided. Using defaults and environment variables.")
        }
      }

      # Handle output directory
      # Priority: output_dir_override > YAML output_dir > default dynamic
      if (!is.null(output_dir_override)) {
        self$output_dir <- output_dir_override
        log_info("Output directory overridden by command line: {self$output_dir}")
      } else if (!is.null(config_path) && file.exists(config_path) && !is.null(yaml::read_yaml(config_path)$output_dir)) {
        # Already set from YAML if it existed
        log_info("Output directory set from YAML: {self$output_dir}")
      } else {
         # Default dynamic output directory if not set by yaml or override
         timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
         self$output_dir <- file.path(paste0("output_revisione_paper_", timestamp)) # Use file.path for OS safety
         log_info("Output directory set to default dynamic: {self$output_dir}")
      }

      # Ensure the output directory exists
      if (!dir.exists(self$output_dir)) {
        dir.create(self$output_dir, recursive = TRUE)
        log_info("Created output directory: {self$output_dir}")
      }

      self$validate()
    },

    validate = function() {
      if (is.null(self$api_key) || self$api_key == "") {
        log_error("API key not configured. Set OPENAI_API_KEY environment variable or provide in config YAML.")
        stop("API key not configured.")
      }
      # Add other validations as needed (e.g., model names are valid)
      # For now, just checking API key.
      log_info("Configuration validated successfully. API key is present.")
      return(TRUE)
    }
  )
)
