# ==== Libraries ====
library(coin)
library(readr)
library(tidyr)
library(dplyr)
library(purrr)
library(stringr)
library(ggplot2)
library(ARTool)

# ==== paths & sets ==== 
setwd("E:/A-Horace/PhD/PANSS-prediction/Final")

langs <- c("cs","de","en","es","escl","fr","gsw","nl","tr","zh")
items <- c("P1","P2","P3","N1","N4","N6","G5","G9")
tasks <- c("FreeSpeech","Dream","Picture","Reading","Recall")

dir.create("results/Figure_3", recursive = TRUE, showWarnings = FALSE)

# functions
add_sig <- function(p){
  ifelse(p < 0.001, "***",
         ifelse(p < 0.01,  "**",
                ifelse(p < 0.05,  "*", "")))
}

# ==== correlations with age ====
# reading data
par_scores_list <- map(items, function(it){
  df <- read_csv(file.path("results/model_eval/par_predicts", paste0(it, ".csv")),
                 show_col_types = FALSE)
  diffs <- df[[paste0("PANSS_", it)]] - df[[paste0("Pred_", it)]]
  df[[paste0("AE_", it)]] <- abs(diffs)
  df$Item <- it
  df
})
names(par_scores_list) <- items

# correlations with age
age_corrs <- map_dfr(items, function(it){
  df <- par_scores_list[[it]]
  x  <- df[[paste0("AE_", it)]]
  y  <- df$Age
  ok <- is.finite(x) & is.finite(y)
  x  <- x[ok]; y <- y[ok]
  ct <- cor.test(x, y, method = "spearman", exact = FALSE)
  tibble(Item = it, r = unname(ct$estimate), p_unc = ct$p.value, n = length(x))
}) %>%
  mutate(p_fdr = p.adjust(p_unc, method = "BH"),
         signif = p_fdr < 0.05)

age_sig_map <- age_corrs %>%
  transmute(Item, siglab = add_sig(p_fdr))

age_corrs <- age_corrs %>%
  arrange(factor(Item, levels = items)) %>%
  mutate(letter = letters[seq_along(Item)],
         title = sprintf("(%s) %s: r=%.3f, q=%.3f%s",
                         letter, Item, r, p_fdr, add_sig(p_fdr)))
# plot
plot_df <- bind_rows(par_scores_list) %>%
  dplyr::select(Item, Age, dplyr::starts_with("AE_")) %>%
  tidyr::pivot_longer(
    cols = dplyr::starts_with("AE_"),
    names_to = "AE_name",
    values_to = "AE"
  ) %>%
  dplyr::filter(stringr::str_remove(AE_name, "^AE_") == Item) %>%
  dplyr::select(Item, Age, AE) %>%
  dplyr::left_join(dplyr::select(age_corrs, Item, title), by = "Item")

plot_df$Item <- factor(plot_df$Item, levels = items)

age_p <- ggplot(plot_df, aes(x = Age, y = AE)) +
  geom_point(color = "#3182bd", alpha = 0.7, size = 1.8) +
  geom_smooth(method = "lm", se = FALSE, color = "#f08787", linewidth = 1.2) +
  facet_wrap(~ title, scales = "free_y") +
  labs(x = "Age", y = "Absolute Error (AE)",
       # title = "Correlations between prediction errors and age"
       ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92"),
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
  )

print(age_p)

ggsave("results/model_eval/bias_check/age.svg", age_p, width = 9, height = 9)
ggsave("results/Figure_3/age.svg", age_p, width = 9, height = 9)

# ==== correlations with education ====
# reading data
par_scores_list <- map(items, function(it){
  df <- read_csv(file.path("results/model_eval/par_predicts", paste0(it, ".csv")),
                 show_col_types = FALSE)
  diffs <- df[[paste0("PANSS_", it)]] - df[[paste0("Pred_", it)]]
  df[[paste0("AE_", it)]] <- abs(diffs)
  df$Item <- it
  df
})
names(par_scores_list) <- items

# correlations
edu_corrs <- map_dfr(items, function(it){
  df <- par_scores_list[[it]]
  x  <- df[[paste0("AE_", it)]]
  y  <- df$Edu
  ok <- is.finite(x) & is.finite(y)
  x  <- x[ok]; y <- y[ok]
  ct <- cor.test(x, y, method = "spearman", exact = FALSE)
  tibble(Item = it, r = unname(ct$estimate), p_unc = ct$p.value, n = length(x))
}) %>%
  mutate(p_fdr = p.adjust(p_unc, method = "BH"),
         signif = p_fdr < 0.05)

edu_sig_map <- edu_corrs %>%
  transmute(Item, siglab = add_sig(p_fdr))

edu_corrs <- edu_corrs %>%
  arrange(factor(Item, levels = items)) %>%
  mutate(letter = letters[seq_along(Item)],
         title = sprintf("(%s) %s: r=%.3f, q=%.3f%s",
                         letter, Item, r, p_fdr, add_sig(p_fdr)))

# plot
plot_df <- bind_rows(par_scores_list) %>%
  dplyr::select(Item,  Edu, dplyr::starts_with("AE_")) %>%
  tidyr::pivot_longer(
    cols = dplyr::starts_with("AE_"),
    names_to = "AE_name",
    values_to = "AE"
  ) %>%
  dplyr::filter(stringr::str_remove(AE_name, "^AE_") == Item) %>%
  dplyr::select(Item,  Edu, AE) %>%
  dplyr::left_join(dplyr::select(edu_corrs, Item, title), by = "Item")

plot_df$Item <- factor(plot_df$Item, levels = items)

edu_p <- ggplot(plot_df, aes(x = Edu, y = AE)) +
  geom_point(color = "#3182bd", alpha = 0.7, size = 1.8) +
  geom_smooth(method = "lm", se = FALSE, color = "#f08787", linewidth = 1.2) +
  facet_wrap(~ title, scales = "free_y") +
  labs(x = "Education", y = "Absolute Error (AE)",
       # title = "Correlations between prediction errors and education"
       ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92"),
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5)
  )

print(edu_p)

ggsave("results/model_eval/bias_check/edu.svg", edu_p, width = 9, height = 9)
ggsave("results/Figure_3/edu.svg", edu_p, width = 9, height = 9)

# ==== correlations with severity ====
# reading data
par_scores_list <- map(items, function(it){
  df <- read_csv(file.path("results/model_eval/par_predicts", paste0(it, ".csv")),
                 show_col_types = FALSE)
  diffs <- df[[paste0("PANSS_", it)]] - df[[paste0("Pred_", it)]]
  df[[paste0("AE_", it)]] <- abs(diffs)
  df$Item <- it
  df
})
names(par_scores_list) <- items

# correlations with severity (PANSS scores)
sev_corrs <- map_dfr(items, function(it){
  df <- par_scores_list[[it]]
  x  <- df[[paste0("AE_", it)]]           # 绝对误差
  y  <- df[[paste0("PANSS_", it)]]        # 对应 PANSS 分数
  ok <- is.finite(x) & is.finite(y)
  x  <- x[ok]; y <- y[ok]
  ct <- cor.test(x, y, method = "spearman", exact = FALSE)
  tibble(Item = it, r = unname(ct$estimate), p_unc = ct$p.value, n = length(x))
}) %>%
  mutate(p_fdr = p.adjust(p_unc, method = "BH"),
         signif = p_fdr < 0.05)

sev_sig_map <- sev_corrs %>%
  transmute(Item, siglab = add_sig(p_fdr))

sev_corrs <- sev_corrs %>%
  arrange(factor(Item, levels = items)) %>%
  mutate(letter = letters[seq_along(Item)],
         title = sprintf("(%s) %s: r=%.3f, q=%.3f%s",
                         letter, Item, r, p_fdr, add_sig(p_fdr)))

# plot
plot_df <- dplyr::bind_rows(par_scores_list) %>%
  dplyr::select(Item, dplyr::starts_with("AE_"), dplyr::starts_with("PANSS_")) %>%
  tidyr::pivot_longer(
    cols = dplyr::starts_with("AE_"),
    names_to = "AE_name",
    values_to = "AE"
  ) %>%
  dplyr::filter(stringr::str_remove(AE_name, "^AE_") == Item) %>%
  dplyr::mutate(
    severity = dplyr::case_when(
      Item == "P1" ~ PANSS_P1,
      Item == "P2" ~ PANSS_P2,
      Item == "P3" ~ PANSS_P3,
      Item == "N1" ~ PANSS_N1,
      Item == "N4" ~ PANSS_N4,
      Item == "N6" ~ PANSS_N6,
      Item == "G5" ~ PANSS_G5,
      Item == "G9" ~ PANSS_G9
    )
  ) %>%
  dplyr::select(Item, severity, AE) %>%
  dplyr::left_join(dplyr::select(sev_corrs, Item, title), by = "Item")

plot_df$Item <- factor(plot_df$Item, levels = items)

sev_p <- ggplot(plot_df, aes(x = severity, y = AE)) +
  geom_point(color = "#3182bd", alpha = 0.7, size = 1.8) +
  geom_smooth(method = "lm", se = FALSE, color = "#f08787", linewidth = 1.2) +
  facet_wrap(~ title, scales = "free_y") +
  labs(x = "Severity (PANSS score)", y = "Absolute Error (AE)",
       # title = "Correlations between prediction errors and severity"
       ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92"),
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5)
  )

print(sev_p)

ggsave("results/model_eval/bias_check/severity.svg", sev_p, width = 9, height = 9)
ggsave("results/Figure_3/severity.svg", sev_p, width = 9, height = 9)

# ==== sex comparisons ====
# reading data
par_scores_list <- map(items, function(it){
  df <- read_csv(file.path("results/model_eval/par_predicts", paste0(it, ".csv")),
                 show_col_types = FALSE)
  diffs <- df[[paste0("PANSS_", it)]] - df[[paste0("Pred_", it)]]
  df[[paste0("AE_", it)]] <- abs(diffs)
  df$Item <- it
  df
})
names(par_scores_list) <- items

# data preparation
plot_df_sex <- dplyr::bind_rows(par_scores_list) %>%
  dplyr::select(Item, Sex, dplyr::starts_with("AE_")) %>%
  tidyr::pivot_longer(
    cols = dplyr::starts_with("AE_"),
    names_to = "AE_name",
    values_to = "AE"
  ) %>%
  dplyr::filter(stringr::str_remove(AE_name, "^AE_") == Item) %>%
  dplyr::transmute(
    Item,
    Sex = as.factor(Sex),
    AE
  )


plot_df_sex$Item <- factor(plot_df_sex$Item,
                           levels = c("P1","P2","P3","N1","N4","N6","G5","G9"))

set.seed(42)

# permutation test
perm_res <- dplyr::bind_rows(lapply(levels(plot_df_sex$Item), function(it){
  df_it <- dplyr::filter(plot_df_sex, Item == it)
  ok <- is.finite(df_it$AE) & !is.na(df_it$Sex)
  df_ok <- df_it[ok, , drop = FALSE]

  if (nrow(df_ok) < 3 || length(unique(df_ok$Sex)) < 2) {
    return(dplyr::tibble(Item = it, p_unc = NA_real_, n = nrow(df_ok)))
  }

  pt <- independence_test(AE ~ Sex, data = df_ok,
                          distribution = approximate(nresample = 10000))
  pval_num <- as.numeric(pvalue(pt))  # 关键：转数值

  dplyr::tibble(Item = it, p_unc = pval_num, n = nrow(df_ok))
})) %>%
  dplyr::mutate(
    p_fdr = p.adjust(p_unc, method = "BH"),
    siglb = add_sig(p_fdr)
  )

annot_df <- plot_df_sex %>%
  dplyr::group_by(Item) %>%
  dplyr::summarise(y = max(AE, na.rm = TRUE), .groups = "drop") %>%
  dplyr::left_join(perm_res, by = "Item") %>%
  dplyr::mutate(y = y * 1.08)


# plot
sex_p <- ggplot(plot_df_sex, aes(x = Item, y = AE, fill = Sex)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.8, width = 0.7) +
  geom_jitter(aes(color = Sex),
              position = position_jitterdodge(jitter.width = 0.18, dodge.width = 0.7),
              size = 0.9, alpha = 0.45) +
  geom_text(data = annot_df,
            aes(x = Item, y = y, label = siglb),
            inherit.aes = FALSE,
            vjust = 0, fontface = "bold", size = 5) +
  labs(x = "PANSS Item", y = "Absolute Error (AE)",
       # title = "AE vs. Sex (Permutation test per item)",
       fill = "Sex", color = "Sex") +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92"),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

print(sex_p)

ggsave("results/model_eval/bias_check/sex.svg", sex_p, width = 9, height = 9)
ggsave("results/Figure_3/sex.svg", sex_p, width = 9, height = 9)

# ==== task comparisons ====
# reading data
task_scores_list <- map(items, function(it){
  df <- read_csv(file.path("results/model_eval/task_predicts", paste0(it, ".csv")),
                 show_col_types = FALSE)
  diffs <- df[[paste0("PANSS_", it)]] - df[[paste0("Pred_", it)]]
  df[[paste0("AE_", it)]] <- abs(diffs)
  df$Item <- it
  df
})
names(task_scores_list) <- items

# ART
art_stats <- map_dfr(items, function(it){
  df_it <- task_scores_list[[it]] %>%
    transmute(PAR = factor(PAR),
              TaskName = factor(TaskName, levels = tasks),
              AE = .data[[paste0("AE_", it)]]) %>%
    filter(is.finite(AE))
  if (dplyr::n_distinct(df_it$TaskName) < 2 || dplyr::n_distinct(df_it$PAR) < 2) {
    return(tibble(Item = it, Fvalue = NA_real_, p_unc = NA_real_))
  }
  fit <- art(AE ~ TaskName + (1|PAR), data = df_it)
  an  <- anova(fit)
  tibble(Item = it, Fvalue = an$F[1], p_unc = an$`Pr(>F)`[1])
}) %>%
  mutate(p_fdr = p.adjust(p_unc, method = "BH")) %>%
  arrange(factor(Item, levels = items)) %>%
  mutate(letter = letters[seq_len(n())],
         title  = sprintf("(%s) %s: F=%.2f, q=%.3f%s",
                          letter, Item, Fvalue, p_fdr, add_sig(p_fdr)))

plot_task <- dplyr::bind_rows(lapply(items, function(it){
  task_scores_list[[it]] %>%
    dplyr::transmute(
      Item = it,
      TaskName,
      AE = .data[[paste0("AE_", it)]]
    )
})) %>%
  dplyr::mutate(
    Item = factor(Item, levels = items),
    TaskName = factor(TaskName, levels = tasks)
  ) %>%
  dplyr::left_join(
    dplyr::select(art_stats, Item, title),
    by = "Item"
  )


# plot
task_p <- ggplot(plot_task, aes(x = TaskName, y = AE, fill = TaskName)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.75, width = 0.7) +
  geom_jitter(aes(color = TaskName), width = 0.15, alpha = 0.45, size = 0.9) +
  facet_wrap(~ title, scales = "free_y") +
  labs(x = "Task", y = "Absolute Error (AE)",
       # title = "Task differences per PANSS item (ART overall effect)"
       ) +
  theme_minimal(base_size = 14) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "grey92"),
    axis.text.x = element_text(angle = 90, hjust = 1),
    strip.text = element_text(face = "bold", size = 12),
    plot.title = element_text(face = "bold", size = 16, hjust = 0.5)
  ) +
  guides(fill = "none", color = "none")

print(task_p)

ggsave("results/model_eval/bias_check/task.svg", task_p, width = 9, height = 9)
ggsave("results/Figure_3/task.svg", task_p, width = 9, height = 9)














