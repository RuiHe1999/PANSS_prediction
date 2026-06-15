library(readxl)
library(ARTool)
library(ggplot2)
library(dplyr)
library(emmeans)
library(multcomp)
library(tidyr)
library(stringr)
library(purrr)
library(ggpubr)

setwd("E:/A-Horace/PhD/PANSS-prediction/Final")
dir.create("results/Figure_2", recursive = TRUE, showWarnings = FALSE)

# read performance data
file_path <- "results/model_comparison/model_feature_performance.xlsx"
sheet_names <- excel_sheets(file_path)

performance_list <- lapply(sheet_names, function(s) {
  read_excel(file_path, sheet = s)
})
names(performance_list) <- sheet_names

performance <- bind_rows(performance_list, .id = "sheet")

performance$model <- as.factor(performance$model)
performance$feature <- as.factor(performance$feature)
performance$symptom <- as.factor(performance$symptom)

vars <- c("test_rmse", "symptom", "feature", "model")
performance <- performance %>%
  filter(across(all_of(vars), ~ !is.na(.))) %>%
  droplevels()

# ART analysis
fit <- art(test_rmse ~ symptom * feature + (1|model), data = performance)
anova_res <- anova(fit)
anova_res 

# post-hoc
posthoc_symptom <- as.data.frame(art.con(fit, "symptom", adjust = "fdr"))
posthoc_feature <- as.data.frame(art.con(fit, "feature", adjust = "fdr"))
posthoc_interaction <- as.data.frame(art.con(fit, "symptom:feature", adjust = "fdr"))

# write the results into a html file
html_file <- "results/model_comparison/ART_results.html"
fileConn <- file(html_file, open = "w", encoding = "UTF-8")
close(fileConn)
sink(html_file)  

cat('<html><head><meta charset="utf-8"><title>ART Results</title>')
cat('<style>
body {font-family: Calibri, sans-serif;}
table {border-collapse: collapse;}
th, td {border: 1px solid black; padding: 4px;}
td.num {text-align: right;}
</style></head><body>\n')

df_to_html_smart <- function(df, title) {
  cat(sprintf('<h2>%s</h2>\n', title))
  df_copy <- df
  
  num_cols <- sapply(df_copy, is.numeric)
  df_copy[, num_cols] <- lapply(df_copy[, num_cols, drop = FALSE],
                                function(x) sprintf("%.3f", x))
  html_table <- paste(capture.output(print(knitr::kable(df_copy, format = "html"))),
                      collapse = "\n")
  html_table <- gsub('<td>', '<td class="num">', html_table)
  cat(html_table, "\n")  
}

df_to_html_smart(anova_res, "ANOVA Results")
df_to_html_smart(posthoc_symptom, "Post-hoc Symptom")
df_to_html_smart(posthoc_feature, "Post-hoc Feature")
df_to_html_smart(posthoc_interaction, "Post-hoc Symptom:Feature Interaction")

cat('</body></html>')
sink()

# plot
performance$symptom <- factor(performance$symptom,
                              levels = c("P1", "P2", "P3", "N1", "N4", "N6", "G5", "G9"))
performance$feature <- factor(performance$feature,
                              levels = c("AcouPros", "m-HuBERT", "Concat"))

p <- ggplot(performance, aes(x = symptom, y = test_rmse, fill = feature)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(aes(color = feature),
              position = position_jitterdodge(jitter.width = 0.2, dodge.width = 0.8),
              size = 1, alpha = 0.6) +
  labs(x = NULL, y = "RMSE", fill = "Feature",) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 0, hjust = 1),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", hjust = 0.5)
  ) +
  coord_cartesian(ylim = c(0.5, 3))

p

ggsave("results/model_comparison/RMSE_boxplot.svg", p, width = 12, height = 5.2)
ggsave("results/Figure_2/A_RMSE_boxplot.svg", p, width = 12, height = 5.2)

# plot the main effect of features
df_hm <- posthoc_feature %>%
  mutate(contrast = gsub("\\(|\\)", "", contrast)) %>%
  separate(contrast, into = c("A","B"), sep = " - ") %>%
  mutate(
    sig = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      TRUE ~ ""
    ),
    label = sprintf("%.3f\n%s", estimate, sig)
  )

feats <- c("AcouPros", "Concat", "m-HuBERT")
df_hm$A <- factor(df_hm$A, levels = feats)
df_hm$B <- factor(df_hm$B, levels = feats)

grid_all <- expand.grid(A = feats, B = feats) %>%
  mutate(A = factor(A, levels = feats),
         B = factor(B, levels = feats))

ia <- as.integer(df_hm$A)
ib <- as.integer(df_hm$B)
swap <- ia > ib

pair_map <- df_hm %>%
  mutate(
    A = ifelse(swap, as.character(B), as.character(A)),
    B = ifelse(swap, as.character(A), as.character(B)),
    estimate = ifelse(swap, -estimate, estimate)
  ) %>%
  mutate(A = factor(A, levels = feats),
         B = factor(B, levels = feats)) %>%
  distinct(A, B, .keep_all = TRUE)

grid_upper <- expand.grid(A = feats, B = feats) %>%
  mutate(A = factor(A, levels = feats),
         B = factor(B, levels = feats)) %>%
  dplyr::filter(as.integer(A) <= as.integer(B))

plot_dat <- grid_upper %>%
  left_join(pair_map %>% select(A, B, estimate, sig), by = c("A","B")) %>%
  mutate(
    sig   = ifelse(A == B, "", sig),
    label = ifelse(A == B | is.na(estimate), "", sprintf("%.3f\n%s", estimate, sig))
  )

grid_all$B <- factor(grid_all$B, levels = rev(feats))
plot_dat$B <- factor(plot_dat$B, levels = rev(feats))

L <- max(abs(plot_dat$estimate), na.rm = TRUE)

ph_feat <- ggplot() +
  geom_tile(data = plot_dat, aes(A, B, fill = estimate), color = "white") +
  geom_text(data = dplyr::filter(plot_dat, label != ""), 
            aes(A, B, label = label), size = 5, lineheight = 0.9) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-L, L), name = "Estimate",
                       na.value = "white") +
  coord_fixed() +
  theme_minimal(base_size = 14) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold", hjust = 0.5)
  ) +
  labs(fill = "Estimate", x = NULL, y = NULL)

ph_feat

ggsave("results/model_comparison/posthoc_feature.svg", ph_feat, width = 5, height = 5)
ggsave("results/Figure_2/B_posthoc_feature.svg", ph_feat, width = 5, height = 5)

# plot the main effect of symptoms
df_hm <- posthoc_symptom %>%
  mutate(contrast = gsub("\\(|\\)", "", contrast)) %>%
  separate(contrast, into = c("A","B"), sep = " - ") %>%
  mutate(
    sig = case_when(
      p.value < 0.001 ~ "***",
      p.value < 0.01  ~ "**",
      p.value < 0.05  ~ "*",
      TRUE ~ ""
    ),
    label = sprintf("%.3f\n%s", estimate, sig)
  )

symptoms <- c("P1", "P2", "P3", "N1", "N4", "N6", "G5", "G9")
df_hm$A <- factor(df_hm$A, levels = symptoms)
df_hm$B <- factor(df_hm$B, levels = symptoms)

grid_all <- expand.grid(A = symptoms, B = symptoms) %>%
  mutate(A = factor(A, levels = symptoms),
         B = factor(B, levels = symptoms))

ia <- as.integer(df_hm$A)
ib <- as.integer(df_hm$B)
swap <- ia > ib

pair_map <- df_hm %>%
  mutate(
    A2 = ifelse(swap, as.character(B), as.character(A)),
    B2 = ifelse(swap, as.character(A), as.character(B)),
    estimate = ifelse(swap, -estimate, estimate)
  ) %>%
  transmute(
    A = factor(A2, levels = symptoms),
    B = factor(B2, levels = symptoms),
    estimate, p.value, sig
  ) %>%
  distinct(A, B, .keep_all = TRUE)

grid_upper <- expand.grid(A = symptoms, B = symptoms) %>%
  mutate(A = factor(A, levels = symptoms),
         B = factor(B, levels = symptoms)) %>%
  dplyr::filter(as.integer(A) <= as.integer(B))

plot_dat <- grid_upper %>%
  left_join(pair_map %>% select(A, B, estimate, sig), by = c("A","B")) %>%
  mutate(
    sig   = ifelse(A == B, "", sig),
    label = ifelse(A == B | is.na(estimate), "", sprintf("%.3f\n%s", estimate, sig))
  )

plot_dat <- grid_upper %>%
  left_join(pair_map %>% select(A, B, estimate, sig), by = c("A","B")) %>%
  mutate(
    sig   = ifelse(A == B, "", sig),
    label = ifelse(A == B | is.na(estimate), "", sprintf("%.3f\n%s", estimate, sig))
  )

grid_all$B <- factor(grid_all$B, levels = rev(symptoms))
plot_dat$B <- factor(plot_dat$B, levels = rev(symptoms))

L <- max(abs(plot_dat$estimate), na.rm = TRUE)

ph_symp <- ggplot() +
  geom_tile(data = plot_dat, aes(A, B, fill = estimate), color = "white") +
  geom_text(data = dplyr::filter(plot_dat, label != ""),  
            aes(A, B, label = label), size = 4.2, lineheight = 0.9) +
  scale_fill_gradient2(low = "blue", mid = "white", high = "red",
                       midpoint = 0, limits = c(-L, L), name = "Estimate",
                       na.value = "white") +
  coord_fixed() +
  theme_minimal(base_size = 14) +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    panel.grid = element_blank(),
    plot.title = element_text(face = "bold", hjust = 0.5)
  ) +
  labs(x = NULL, y = NULL)

ph_symp


ggsave("results/model_comparison/posthoc_symptom.svg", ph_symp, width = 7, height = 7)
ggsave("results/Figure_2/C_posthoc_symptom.svg", ph_symp, width = 7, height = 7)
