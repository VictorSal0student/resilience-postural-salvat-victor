# ============================================================================
# Statistical Analysis for Locomotor Resilience Data
# ============================================================================
# 
# This script performs statistical analyses on resilience metrics:
# - Repeated measures ANOVA (Group × Perturbation)
# - Post-hoc tests (Tukey HSD, pairwise comparisons)
# - Normality and homoscedasticity diagnostics
# - Effect size calculations
# - Non-parametric alternatives (if needed)
#
# Author: Victor SALVAT
# Date: 2026-03-30
# Project: Master IEAP - Locomotor Resilience Analysis
# ============================================================================

# Load required libraries
library(tidyverse)      # Data manipulation and visualization
library(car)            # ANOVA diagnostics (Levene's test)
library(rstatix)        # Pipe-friendly stats functions
library(emmeans)        # Estimated marginal means
library(effectsize)     # Effect size calculations
library(ggpubr)         # Publication-ready plots

# ============================================================================
# 1. DATA LOADING
# ============================================================================

# Load aggregated metrics
data <- read.csv("results/metrics/group/all_participants.csv", 
                 stringsAsFactors = TRUE)

# Display structure
cat("\n=== DATA STRUCTURE ===\n")
str(data)
cat("\n=== FIRST ROWS ===\n")
head(data)

# Check for missing values
cat("\n=== MISSING VALUES ===\n")
colSums(is.na(data))

# Summary statistics
cat("\n=== SUMMARY STATISTICS ===\n")
summary(data)

# ============================================================================
# 2. DATA PREPARATION
# ============================================================================

# Convert to long format for repeated measures
data_long <- data %>%
  pivot_longer(
    cols = starts_with("Slow_") | starts_with("Fast_"),
    names_to = c("Perturbation", "Metric"),
    names_pattern = "(.*)_(.*)",
    values_to = "Value"
  ) %>%
  mutate(
    Participant = as.factor(Participant),
    Group = as.factor(Group),
    Perturbation = as.factor(Perturbation),
    Metric = as.factor(Metric)
  )

# Separate datasets for each metric
recovery_data <- data_long %>% 
  filter(Metric == "RecoveryTime") %>%
  select(Participant, Group, Perturbation, Value) %>%
  rename(RecoveryTime = Value)

peak_data <- data_long %>% 
  filter(Metric == "PeakDeviation") %>%
  select(Participant, Group, Perturbation, Value) %>%
  rename(PeakDeviation = Value)

cat("\n=== DATA PREPARATION COMPLETE ===\n")
cat("Recovery Time observations:", nrow(recovery_data), "\n")
cat("Peak Deviation observations:", nrow(peak_data), "\n")

# ============================================================================
# 3. DESCRIPTIVE STATISTICS
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("DESCRIPTIVE STATISTICS - RECOVERY TIME\n")
cat("="*70 %+% "\n")

recovery_summary <- recovery_data %>%
  group_by(Group, Perturbation) %>%
  summarise(
    N = n(),
    Mean = mean(RecoveryTime, na.rm = TRUE),
    SD = sd(RecoveryTime, na.rm = TRUE),
    SE = SD / sqrt(N),
    Median = median(RecoveryTime, na.rm = TRUE),
    Min = min(RecoveryTime, na.rm = TRUE),
    Max = max(RecoveryTime, na.rm = TRUE),
    .groups = 'drop'
  )

print(recovery_summary)

cat("\n" %+% "="*70 %+% "\n")
cat("DESCRIPTIVE STATISTICS - PEAK DEVIATION\n")
cat("="*70 %+% "\n")

peak_summary <- peak_data %>%
  group_by(Group, Perturbation) %>%
  summarise(
    N = n(),
    Mean = mean(PeakDeviation, na.rm = TRUE),
    SD = sd(PeakDeviation, na.rm = TRUE),
    SE = SD / sqrt(N),
    Median = median(PeakDeviation, na.rm = TRUE),
    Min = min(PeakDeviation, na.rm = TRUE),
    Max = max(PeakDeviation, na.rm = TRUE),
    .groups = 'drop'
  )

print(peak_summary)

# Export summary tables
write.csv(recovery_summary, "results/metrics/group/recovery_time_summary.csv", row.names = FALSE)
write.csv(peak_summary, "results/metrics/group/peak_deviation_summary.csv", row.names = FALSE)

# ============================================================================
# 4. ASSUMPTION TESTING
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("STATISTICAL ASSUMPTIONS - RECOVERY TIME\n")
cat("="*70 %+% "\n")

# --- Normality (Shapiro-Wilk test) ---
cat("\n[1] Normality Tests (Shapiro-Wilk):\n")

normality_recovery <- recovery_data %>%
  group_by(Group, Perturbation) %>%
  shapiro_test(RecoveryTime)

print(normality_recovery)

if(any(normality_recovery$p < 0.05)) {
  cat("\n⚠ WARNING: Some groups violate normality assumption (p < 0.05)\n")
  cat("   Consider non-parametric tests or data transformation\n")
} else {
  cat("\n✓ Normality assumption met for all groups\n")
}

# --- Homogeneity of Variance (Levene's test) ---
cat("\n[2] Homogeneity of Variance (Levene's test):\n")

levene_recovery <- recovery_data %>%
  levene_test(RecoveryTime ~ Group * Perturbation)

print(levene_recovery)

if(levene_recovery$p < 0.05) {
  cat("\n⚠ WARNING: Homogeneity of variance assumption violated (p < 0.05)\n")
} else {
  cat("\n✓ Homogeneity of variance assumption met\n")
}

# --- QQ Plots ---
cat("\n[3] Generating QQ plots...\n")

pdf("results/figures/qq_plots_recovery.pdf", width = 10, height = 6)
par(mfrow = c(2, 2))

for (grp in levels(recovery_data$Group)) {
  for (pert in levels(recovery_data$Perturbation)) {
    subset_data <- recovery_data %>%
      filter(Group == grp, Perturbation == pert) %>%
      pull(RecoveryTime)
    
    qqnorm(subset_data, main = paste(grp, "-", pert))
    qqline(subset_data, col = "red")
  }
}

dev.off()
cat("   Saved: results/figures/qq_plots_recovery.pdf\n")

# ============================================================================
# 5. REPEATED MEASURES ANOVA - RECOVERY TIME
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("REPEATED MEASURES ANOVA - RECOVERY TIME\n")
cat("="*70 %+% "\n")

# Fit ANOVA model
anova_recovery <- aov(RecoveryTime ~ Group * Perturbation + Error(Participant/Perturbation),
                      data = recovery_data)

cat("\nANOVA Results:\n")
summary(anova_recovery)

# Alternative: Using ez package for more detailed output
# Uncomment if ez package is installed
# library(ez)
# anova_ez <- ezANOVA(
#   data = recovery_data,
#   dv = RecoveryTime,
#   wid = Participant,
#   within = Perturbation,
#   between = Group,
#   detailed = TRUE,
#   type = 3
# )
# print(anova_ez)

# Effect sizes (partial eta squared)
cat("\nEffect Sizes (Partial η²):\n")
effectsize_recovery <- eta_squared(anova_recovery, partial = TRUE)
print(effectsize_recovery)

# ============================================================================
# 6. POST-HOC TESTS - RECOVERY TIME
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("POST-HOC TESTS - RECOVERY TIME\n")
cat("="*70 %+% "\n")

# Estimated marginal means
emm_recovery <- emmeans(aov(RecoveryTime ~ Group * Perturbation, data = recovery_data),
                        ~ Group | Perturbation)

cat("\nEstimated Marginal Means:\n")
print(emm_recovery)

# Pairwise comparisons with Bonferroni correction
cat("\nPairwise Comparisons (Bonferroni corrected):\n")
pairs_recovery <- pairs(emm_recovery, adjust = "bonferroni")
print(pairs_recovery)

# Tukey HSD (if main effects significant)
cat("\nTukey HSD:\n")
tukey_recovery <- TukeyHSD(aov(RecoveryTime ~ Group * Perturbation, data = recovery_data))
print(tukey_recovery)

# Export post-hoc results
capture.output(
  summary(pairs_recovery),
  file = "results/metrics/group/posthoc_recovery.txt"
)

# ============================================================================
# 7. REPEATED MEASURES ANOVA - PEAK DEVIATION
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("REPEATED MEASURES ANOVA - PEAK DEVIATION\n")
cat("="*70 %+% "\n")

# Fit ANOVA model
anova_peak <- aov(PeakDeviation ~ Group * Perturbation + Error(Participant/Perturbation),
                  data = peak_data)

cat("\nANOVA Results:\n")
summary(anova_peak)

# Effect sizes
cat("\nEffect Sizes (Partial η²):\n")
effectsize_peak <- eta_squared(anova_peak, partial = TRUE)
print(effectsize_peak)

# Post-hoc
emm_peak <- emmeans(aov(PeakDeviation ~ Group * Perturbation, data = peak_data),
                    ~ Group | Perturbation)

cat("\nPairwise Comparisons (Bonferroni corrected):\n")
pairs_peak <- pairs(emm_peak, adjust = "bonferroni")
print(pairs_peak)

# ============================================================================
# 8. NON-PARAMETRIC ALTERNATIVE (if assumptions violated)
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("NON-PARAMETRIC TESTS (Friedman + Wilcoxon)\n")
cat("="*70 %+% "\n")

# Friedman test (non-parametric alternative to repeated measures ANOVA)
cat("\nFriedman Test - Recovery Time:\n")
friedman_recovery <- recovery_data %>%
  friedman_test(RecoveryTime ~ Perturbation | Participant)
print(friedman_recovery)

# Wilcoxon signed-rank test for pairwise comparisons
if(friedman_recovery$p < 0.05) {
  cat("\nPairwise Wilcoxon tests:\n")
  wilcox_recovery <- recovery_data %>%
    wilcox_test(RecoveryTime ~ Perturbation, paired = TRUE) %>%
    adjust_pvalue(method = "bonferroni")
  print(wilcox_recovery)
}

# ============================================================================
# 9. EXPORT RESULTS
# ============================================================================

cat("\n" %+% "="*70 %+% "\n")
cat("EXPORTING RESULTS\n")
cat("="*70 %+% "\n")

# Create comprehensive results table
results_table <- data.frame(
  Metric = c("Recovery Time", "Peak Deviation"),
  ANOVA_p = c(
    summary(anova_recovery)[[2]][[1]][["Pr(>F)"]][1],
    summary(anova_peak)[[2]][[1]][["Pr(>F)"]][1]
  ),
  Effect_Size = c(
    effectsize_recovery$Eta2_partial[1],
    effectsize_peak$Eta2_partial[1]
  )
)

write.csv(results_table, "results/metrics/group/statistical_summary.csv", row.names = FALSE)

cat("\n✓ Statistical analysis complete!\n")
cat("  Summary tables: results/metrics/group/\n")
cat("  Post-hoc results: results/metrics/group/posthoc_recovery.txt\n")
cat("  QQ plots: results/figures/qq_plots_recovery.pdf\n")

cat("\n" %+% "="*70 %+% "\n")
cat("ANALYSIS FINISHED\n")
cat("="*70 %+% "\n")

# ============================================================================
# END OF SCRIPT
# ============================================================================
