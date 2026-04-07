# ── plot_theme.R ──────────────────────────────────────────────────────────────
# Shared ggplot2 theme and color palette for all resilience figures

theme_resilience <- function(base_size = 13) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title      = element_text(face = "bold"),
      legend.position = "bottom"
    )
}

group_colors <- list(
  fill  = c("Young" = "#2196F3", "Aging" = "#FF9800"),
  color = c("Young" = "#1565C0", "Aging" = "#E65100")
)
