# Interactive 3D Torus Visualization Functions
# Author: Victor SALVAT
# Date: 2026-04-07
# 
# Usage in Rmd:
#   source("sources/r/torus_3d.R")
#   plot_torus_3d("results/figures/phase_space_SLOW.csv", "State-Space 3D — Slow Perturbation")

library(plotly)
library(readr)
library(dplyr)

plot_torus_3d <- function(csv_path, title = "State-Space 3D") {
  
  if (!file.exists(csv_path)) {
    stop(sprintf("File not found: %s", csv_path))
  }
  
  data <- read_csv(csv_path, show_col_types = FALSE)
  
  thresholds_path <- gsub("\\.csv$", "_thresholds.csv", csv_path)
  thresholds <- read_csv(thresholds_path, show_col_types = FALSE)
  T1_value <- thresholds$Value[thresholds$Threshold == "T1"]
  
  traj_color <- ifelse(grepl("SLOW", csv_path, ignore.case = TRUE), 
                       "#ff6b35",
                       "#d62828")
  traj_name <- ifelse(grepl("SLOW", csv_path, ignore.case = TRUE),
                      "Slow Perturbation",
                      "Fast Perturbation")
  
  fig <- plot_ly()
  
  data_T1 <- data[data$Stability == "T1", ]
  if (nrow(data_T1) > 0) {
    fig <- fig %>% add_trace(
      data = data_T1,
      x = ~X, y = ~Y, z = ~Z,
      type = "scatter3d",
      mode = "lines",
      line = list(color = "#757575", width = 2),
      name = "T1 Stability Zone",
      hoverinfo = "name",
      showlegend = TRUE
    )
  }
  
  fig <- fig %>% add_trace(
    data = data,
    x = ~X_ref, y = ~Y_ref, z = ~Z_ref,
    type = "scatter3d",
    mode = "lines",
    line = list(color = "#000000", width = 2.5),
    name = "Reference Trajectory",
    hoverinfo = "name",
    showlegend = TRUE
  )
  
  fig <- fig %>% add_trace(
    data = data,
    x = ~X, y = ~Y, z = ~Z,
    type = "scatter3d",
    mode = "lines",
    line = list(color = traj_color, width = 2),
    text = ~paste0(
      "Time: ", round(Time, 2), " s<br>",
      "Distance: ", round(Distance, 2), " mm<br>",
      "Stability: ", Stability
    ),
    hoverinfo = "text",
    name = traj_name,
    showlegend = TRUE
  )
  
  fig <- fig %>% layout(
    title = list(
      text = title,
      font = list(size = 14, family = "Times New Roman", color = "#000000")
    ),
    scene = list(
      xaxis = list(
        title = list(text = "X (mm)", font = list(size = 12, family = "Times New Roman")),
        gridcolor = "#e0e0e0",
        showbackground = TRUE,
        backgroundcolor = "#ffffff",
        zerolinecolor = "#000000"
      ),
      yaxis = list(
        title = list(text = "Y (mm)", font = list(size = 12, family = "Times New Roman")),
        gridcolor = "#e0e0e0",
        showbackground = TRUE,
        backgroundcolor = "#ffffff",
        zerolinecolor = "#000000"
      ),
      zaxis = list(
        title = list(text = "Z (mm)", font = list(size = 12, family = "Times New Roman")),
        gridcolor = "#e0e0e0",
        showbackground = TRUE,
        backgroundcolor = "#ffffff",
        zerolinecolor = "#000000"
      ),
      camera = list(
        eye = list(x = 1.5, y = 1.5, z = 1.2)
      ),
      aspectmode = "cube"
    ),
    showlegend = TRUE,
    legend = list(
      x = 0.02, y = 0.98,
      bgcolor = "rgba(255,255,255,0.95)",
      bordercolor = "#000000",
      borderwidth = 1,
      font = list(size = 11, family = "Times New Roman", color = "#000000")
    ),
    paper_bgcolor = "#ffffff",
    plot_bgcolor = "#ffffff"
  ) %>%
    config(
      displayModeBar = TRUE,
      displaylogo = FALSE,
      modeBarButtonsToRemove = c("sendDataToCloud", "lasso2d", "select2d")
    )
  
  return(fig)
}
