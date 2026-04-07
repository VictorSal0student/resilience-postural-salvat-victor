# ============================================================================
# Group Comparison Analysis - Young vs Aging
# ============================================================================
# 
# This script focuses on comparing locomotor resilience between age groups:
# - Descriptive statistics by group
# - Independent t-tests
# - Cohen's d effect sizes
# - Publication-ready visualizations
#
# Author: Victor SALVAT
# Date: 2026-03-30
# Project: Master IEAP - Locomotor Resilience Analysis
# ============================================================================

# Load required libraries
library(tidyverse)
library(ggpubr)
library(rstatix)
library(effectsize)
library(cowplot)

# Set plotting theme
theme_set(theme_pubr())

# ============================================================================
# 1. DATA LOADING
# ============================================================================

cat("\n=== LOADING DATA ===\n")

data <- read.csv("results/metrics/group/all_participants.csv", 
                 stringsAsFactors = TRUE)

cat("Participants loaded:", nrow(data), "\n")
cat("Groups:", levels(data$Group), "\n")

# Group counts
group_counts <- table(data$Group)
print(group_counts)

# ============================================================================
# 2. YOUNG vs AGING - RECOVERY TIME
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("GROUP COMPARISON - RECOVERY TIME\n")
cat("="*70 %+% "\n")

# --- Slow Perturbation ---
cat("\n[SLOW PERTURBATION]\n")

cat("\nDescriptive Statistics:\n")
slow_recovery_summary <- data %>%
  group_by(Group) %>%
  summarise(
    N = n(),
    Mean = mean(Slow_RecoveryTime, na.rm = TRUE),
    SD = sd(Slow_RecoveryTime, na.rm = TRUE),
    SE = SD / sqrt(N),
    Median = median(Slow_RecoveryTime, na.rm = TRUE),
    IQR = IQR(Slow_RecoveryTime, na.rm = TRUE),
    .groups = 'drop'
  )
print(slow_recovery_summary)

# Independent t-test
cat("\nIndependent t-test:\n")
slow_ttest <- t.test(Slow_RecoveryTime ~ Group, data = data, var.equal = FALSE)
print(slow_ttest)

# Effect size (Cohen's d)
cat("\nEffect Size (Cohen's d):\n")
slow_d <- cohens_d(Slow_RecoveryTime ~ Group, data = data)
print(slow_d)

# --- Fast Perturbation ---
cat("\n[FAST PERTURBATION]\n")

cat("\nDescriptive Statistics:\n")
fast_recovery_summary <- data %>%
  group_by(Group) %>%
  summarise(
    N = n(),
    Mean = mean(Fast_RecoveryTime, na.rm = TRUE),
    SD = sd(Fast_RecoveryTime, na.rm = TRUE),
    SE = SD / sqrt(N),
    Median = median(Fast_RecoveryTime, na.rm = TRUE),
    IQR = IQR(Fast_RecoveryTime, na.rm = TRUE),
    .groups = 'drop'
  )
print(fast_recovery_summary)

# Independent t-test
cat("\nIndependent t-test:\n")
fast_ttest <- t.test(Fast_RecoveryTime ~ Group, data = data, var.equal = FALSE)
print(fast_ttest)

# Effect size
cat("\nEffect Size (Cohen's d):\n")
fast_d <- cohens_d(Fast_RecoveryTime ~ Group, data = data)
print(fast_d)

# ============================================================================
# 3. YOUNG vs AGING - PEAK DEVIATION
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("GROUP COMPARISON - PEAK DEVIATION\n")
cat("="*70 %+% "\n")

# --- Slow Perturbation ---
cat("\n[SLOW PERTURBATION]\n")

cat("\nDescriptive Statistics:\n")
slow_peak_summary <- data %>%
  group_by(Group) %>%
  summarise(
    N = n(),
    Mean = mean(Slow_PeakDeviation, na.rm = TRUE),
    SD = sd(Slow_PeakDeviation, na.rm = TRUE),
    SE = SD / sqrt(N),
    Median = median(Slow_PeakDeviation, na.rm = TRUE),
    IQR = IQR(Slow_PeakDeviation, na.rm = TRUE),
    .groups = 'drop'
  )
print(slow_peak_summary)

# t-test
slow_peak_ttest <- t.test(Slow_PeakDeviation ~ Group, data = data)
print(slow_peak_ttest)

# Effect size
slow_peak_d <- cohens_d(Slow_PeakDeviation ~ Group, data = data)
print(slow_peak_d)

# --- Fast Perturbation ---
cat("\n[FAST PERTURBATION]\n")

fast_peak_summary <- data %>%
  group_by(Group) %>%
  summarise(
    N = n(),
    Mean = mean(Fast_PeakDeviation, na.rm = TRUE),
    SD = sd(Fast_PeakDeviation, na.rm = TRUE),
    SE = SD / sqrt(N),
    Median = median(Fast_PeakDeviation, na.rm = TRUE),
    IQR = IQR(Fast_PeakDeviation, na.rm = TRUE),
    .groups = 'drop'
  )
print(fast_peak_summary)

# t-test
fast_peak_ttest <- t.test(Fast_PeakDeviation ~ Group, data = data)
print(fast_peak_ttest)

# Effect size
fast_peak_d <- cohens_d(Fast_PeakDeviation ~ Group, data = data)
print(fast_peak_d)

# ============================================================================
# 4. VISUALIZATIONS
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("GENERATING VISUALIZATIONS\n")
cat("="*70 %+% "\n")

# Prepare data for plotting
data_long <- data %>%
  pivot_longer(
    cols = c(Slow_RecoveryTime, Fast_RecoveryTime),
    names_to = "Perturbation",
    values_to = "RecoveryTime"
  ) %>%
  mutate(
    Perturbation = recode(Perturbation, 
                          "Slow_RecoveryTime" = "Slow",
                          "Fast_RecoveryTime" = "Fast")
  )

# --- Plot 1: Recovery Time Comparison ---
p1 <- ggplot(data_long, aes(x = Perturbation, y = RecoveryTime, fill = Group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.2),
              alpha = 0.5, size = 2) +
  stat_compare_means(aes(group = Group), 
                     label = "p.format",
                     method = "t.test") +
  scale_fill_manual(values = c("Young" = "#4A90E2", "Aging" = "#E94B3C")) +
  labs(
    title = "Recovery Time - Young vs Aging",
    x = "Perturbation Type",
    y = "Recovery Time (s)",
    fill = "Group"
  ) +
  theme_pubr() +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold", size = 12)
  )

ggsave("results/figures/group_comparison_recovery_time.png", 
       p1, width = 8, height = 6, dpi = 300)

cat("  ✓ Saved: results/figures/group_comparison_recovery_time.png\n")

# --- Plot 2: Peak Deviation Comparison ---
data_long_peak <- data %>%
  pivot_longer(
    cols = c(Slow_PeakDeviation, Fast_PeakDeviation),
    names_to = "Perturbation",
    values_to = "PeakDeviation"
  ) %>%
  mutate(
    Perturbation = recode(Perturbation,
                          "Slow_PeakDeviation" = "Slow",
                          "Fast_PeakDeviation" = "Fast")
  )

p2 <- ggplot(data_long_peak, aes(x = Perturbation, y = PeakDeviation, fill = Group)) +
  geom_boxplot(alpha = 0.7, outlier.shape = NA) +
  geom_jitter(position = position_jitterdodge(jitter.width = 0.2),
              alpha = 0.5, size = 2) +
  stat_compare_means(aes(group = Group),
                     label = "p.format",
                     method = "t.test") +
  scale_fill_manual(values = c("Young" = "#4A90E2", "Aging" = "#E94B3C")) +
  labs(
    title = "Peak Deviation - Young vs Aging",
    x = "Perturbation Type",
    y = "Peak Deviation (mm)",
    fill = "Group"
  ) +
  theme_pubr() +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold", size = 14),
    axis.title = element_text(face = "bold", size = 12)
  )

ggsave("results/figures/group_comparison_peak_deviation.png",
       p2, width = 8, height = 6, dpi = 300)

cat("  ✓ Saved: results/figures/group_comparison_peak_deviation.png\n")

# --- Plot 3: Combined Figure ---
combined_plot <- plot_grid(p1, p2, ncol = 1, labels = c("A", "B"))

ggsave("results/figures/group_comparison_combined.png",
       combined_plot, width = 10, height = 12, dpi = 300)

cat("  ✓ Saved: results/figures/group_comparison_combined.png\n")

# --- Plot 4: Correlation Slow vs Fast ---
p3 <- ggplot(data, aes(x = Slow_RecoveryTime, y = Fast_RecoveryTime, 
                       color = Group, shape = Group)) +
  geom_point(size = 3, alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, alpha = 0.2) +
  scale_color_manual(values = c("Young" = "#4A90E2", "Aging" = "#E94B3C")) +
  labs(
    title = "Slow vs Fast Recovery Time Correlation",
    x = "Slow Recovery Time (s)",
    y = "Fast Recovery Time (s)",
    color = "Group",
    shape = "Group"
  ) +
  theme_pubr() +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold", size = 14)
  )

ggsave("results/figures/correlation_slow_vs_fast.png",
       p3, width = 8, height = 6, dpi = 300)

cat("  ✓ Saved: results/figures/correlation_slow_vs_fast.png\n")

# ============================================================================
# 5. EXPORT SUMMARY TABLE
# ============================================================================

cat("\n=== EXPORTING SUMMARY TABLE ===\n")

# Combine all statistics
comparison_summary <- data.frame(
  Metric = rep(c("Recovery Time", "Peak Deviation"), each = 2),
  Perturbation = rep(c("Slow", "Fast"), 2),
  Young_Mean = c(
    slow_recovery_summary$Mean[slow_recovery_summary$Group == "Young"],
    fast_recovery_summary$Mean[fast_recovery_summary$Group == "Young"],
    slow_peak_summary$Mean[slow_peak_summary$Group == "Young"],
    fast_peak_summary$Mean[fast_peak_summary$Group == "Young"]
  ),
  Young_SD = c(
    slow_recovery_summary$SD[slow_recovery_summary$Group == "Young"],
    fast_recovery_summary$SD[fast_recovery_summary$Group == "Young"],
    slow_peak_summary$SD[slow_peak_summary$Group == "Young"],
    fast_peak_summary$SD[fast_peak_summary$Group == "Young"]
  ),
  Aging_Mean = c(
    slow_recovery_summary$Mean[slow_recovery_summary$Group == "Aging"],
    fast_recovery_summary$Mean[fast_recovery_summary$Group == "Aging"],
    slow_peak_summary$Mean[slow_peak_summary$Group == "Aging"],
    fast_peak_summary$Mean[fast_peak_summary$Group == "Aging"]
  ),
  Aging_SD = c(
    slow_recovery_summary$SD[slow_recovery_summary$Group == "Aging"],
    fast_recovery_summary$SD[fast_recovery_summary$Group == "Aging"],
    slow_peak_summary$SD[slow_peak_summary$Group == "Aging"],
    fast_peak_summary$SD[fast_peak_summary$Group == "Aging"]
  ),
  t_statistic = c(
    slow_ttest$statistic,
    fast_ttest$statistic,
    slow_peak_ttest$statistic,
    fast_peak_ttest$statistic
  ),
  p_value = c(
    slow_ttest$p.value,
    fast_ttest$p.value,
    slow_peak_ttest$p.value,
    fast_peak_ttest$p.value
  ),
  Cohens_d = c(
    slow_d$Cohens_d,
    fast_d$Cohens_d,
    slow_peak_d$Cohens_d,
    fast_peak_d$Cohens_d
  )
)

# Round numeric columns
comparison_summary <- comparison_summary %>%
  mutate(across(where(is.numeric), ~round(.x, 3)))

print(comparison_summary)

write.csv(comparison_summary, 
          "results/metrics/group/group_comparison_summary.csv",
          row.names = FALSE)

cat("\n✓ Exported: results/metrics/group/group_comparison_summary.csv\n")

# ============================================================================
# 6. INTERPRETATION GUIDE
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("INTERPRETATION GUIDE\n")
cat("="*70 %+% "\n")

cat("\nEffect Size Interpretation (Cohen's d):\n")
cat("  Small:  d = 0.2\n")
cat("  Medium: d = 0.5\n")
cat("  Large:  d = 0.8\n")

cat("\nStatistical Significance:\n")
cat("  p < 0.05:  Significant\n")
cat("  p < 0.01:  Highly significant\n")
cat("  p < 0.001: Very highly significant\n")

cat("\n" %+% "="*70 %+% "\n")
cat("GROUP COMPARISON ANALYSIS COMPLETE\n")
cat("="*70 %+% "\n")

# ============================================================================
# END OF SCRIPT
# ============================================================================
