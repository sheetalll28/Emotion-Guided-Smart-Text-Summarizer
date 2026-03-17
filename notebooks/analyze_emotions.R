# Install if needed:
# install.packages(c("shiny", "tidyverse", "scales"))

library(shiny)
library(tidyverse)
library(scales)

# ---------------------------
# Load & Clean Data
# ---------------------------

csv_path <- "../data/processed/summarizer_analytics.csv"

df <- read_csv(csv_path, show_col_types = FALSE)

df <- df %>%
  mutate(
    IsInSummary = tolower(IsInSummary) == "true",
    PredictedEmotion = as.factor(PredictedEmotion),
    AudioFileName = as.factor(AudioFileName)
  )

# ---------------------------
# UI
# ---------------------------

ui <- fluidPage(
  
  titlePanel("Emotion & Summary Analytics Dashboard"),
  
  sidebarLayout(
    
    sidebarPanel(
      h4("Filters"),
      
      selectInput(
        "file_filter",
        "Select Audio File:",
        choices = c("All", levels(df$AudioFileName)),
        selected = "All"
      )
    ),
    
    mainPanel(
      
      fluidRow(
        column(6, plotOutput("emotionPlot")),
        column(6, plotOutput("summaryPlot"))
      ),
      
      fluidRow(
        column(6, plotOutput("filePlot")),
        column(6, plotOutput("confidencePlot"))
      )
      
    )
  )
)

# ---------------------------
# SERVER
# ---------------------------

server <- function(input, output) {
  
  # Reactive filtered dataset
  filtered_data <- reactive({
    if (input$file_filter == "All") {
      df
    } else {
      df %>% filter(AudioFileName == input$file_filter)
    }
  })
  
  # ---------------------------
  # Emotion Distribution
  # ---------------------------
  output$emotionPlot <- renderPlot({
    
    filtered_data() %>%
      count(PredictedEmotion) %>%
      ggplot(aes(x = reorder(PredictedEmotion, -n),
                 y = n,
                 fill = PredictedEmotion)) +
      geom_col() +
      theme_minimal() +
      theme(legend.position = "none") +
      labs(
        title = "Emotion Distribution",
        x = "Emotion",
        y = "Sentence Count"
      )
  })
  
  # ---------------------------
  # Summary Inclusion by Emotion
  # ---------------------------
  output$summaryPlot <- renderPlot({
    
    filtered_data() %>%
      group_by(PredictedEmotion) %>%
      summarise(
        Total = n(),
        Included = sum(IsInSummary),
        InclusionRate = Included / Total,
        .groups = "drop"
      ) %>%
      ggplot(aes(x = reorder(PredictedEmotion, -InclusionRate),
                 y = InclusionRate,
                 fill = PredictedEmotion)) +
      geom_col() +
      scale_y_continuous(labels = percent) +
      theme_minimal() +
      theme(legend.position = "none") +
      labs(
        title = "Summary Inclusion Rate by Emotion",
        y = "Inclusion Rate"
      )
  })
  
  # ---------------------------
  # Summary Inclusion per File
  # ---------------------------
  output$filePlot <- renderPlot({
    
    df %>%
      group_by(AudioFileName) %>%
      summarise(
        Total = n(),
        Included = sum(IsInSummary),
        InclusionRate = Included / Total,
        .groups = "drop"
      ) %>%
      ggplot(aes(x = reorder(AudioFileName, -InclusionRate),
                 y = InclusionRate,
                 fill = AudioFileName)) +
      geom_col() +
      scale_y_continuous(labels = percent) +
      theme_minimal() +
      theme(legend.position = "none") +
      labs(
        title = "Overall Inclusion Rate per File",
        y = "Inclusion Rate"
      )
  })
  
  # ---------------------------
  # Confidence Analysis (FIXED)
  # ---------------------------
  output$confidencePlot <- renderPlot({
    
    filtered_data() %>%
      ggplot(aes(x = PredictedEmotion,
                 y = EmotionConfidence,
                 fill = PredictedEmotion)) +
      geom_boxplot(alpha = 0.7) +
      theme_minimal() +
      theme(legend.position = "none") +
      labs(
        title = "Model Confidence by Emotion",
        x = "Emotion",
        y = "Confidence Score"
      )
  })
}

# ---------------------------
# Run App
# ---------------------------

shinyApp(ui = ui, server = server)
