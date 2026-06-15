library(dplyr)
library(tidyr)
library(ggplot2)
library(ggrepel)
library(grid)

setwd("E:/A-Horace/PhD/PANSS-prediction/Final")

dir.create("results/Figure_2", recursive = TRUE, showWarnings = FALSE)

# Language labels and order
lang_labels <- c(
  test="Test", cs="Czech", de="German", en="English", es="Spanish",
  escl="Chilean_Spanish", fr="French", gsw="Swiss_German", nl="Dutch", 
  zh="Chinese", tr="Turkish"
)
src_levels <- names(lang_labels)

best_tbl <- readr::read_csv("results/model_comparison/best_model.csv",
                            show_col_types = FALSE) %>%
  dplyr::select(model, feature, symptom)

perf <- readr::read_csv("results/model_comparison/best_model_performance.csv",
                        show_col_types = FALSE)

df_long <- perf %>%
  tidyr::pivot_longer(
    cols = -symptom,
    names_to = c("src", "metric", "method"),
    names_pattern = "(.*)_(rmse|r2)_(seg|par)",
    values_to = "val"
  ) %>%
  dplyr::mutate(
    symptom = factor(symptom, levels = c("P1","P2","P3","N1","N4","N6","G5","G9")),
    src = factor(src, levels = c("test","cs","de","en","es","escl","fr","gsw","nl","tr","zh"))
  )

# Select RMSE + seg
df_rmse_seg <- df_long %>%
  filter(metric == "rmse", method == "seg") %>%
  mutate(
    src     = factor(src, levels = src_levels),
    symptom = factor(symptom, levels = c("P1","P2","P3","N1","N4","N6","G5","G9"))
  )

# Keep only model, feature, and symptom for annotation
best_tbl_min <- best_tbl %>% select(model, feature, symptom)

# Last available point for each line + label text (symptom · model | feature)
end_annot <- df_rmse_seg %>%
  arrange(symptom, src) %>%
  group_by(symptom) %>%
  filter(!is.na(val)) %>%
  slice_tail(n = 1) %>%
  ungroup() %>%
  left_join(best_tbl %>% select(model, feature, symptom), by = "symptom") %>%
  mutate(label = paste0(symptom, " · ", model, " | ", feature))

# X position for labels (outside the plotting area)
x_out <- length(src_levels) + 0.5

# Arrange labels vertically (from high to low)
end_annot <- end_annot %>%
  arrange(desc(val)) %>%
  mutate(
    y_label = seq(from = max(val) + 0.05, to = min(val) - 0.05, length.out = dplyr::n())
  )

rmse_seg <- ggplot(df_rmse_seg, aes(x = src, y = val, group = symptom, color = symptom)) +
  geom_line(linewidth = 1, alpha = 0.9, na.rm = TRUE) +
  geom_point(size = 2.2, alpha = 0.9, na.rm = TRUE) +
  # Highlight Test point
  geom_point(
    data = dplyr::filter(df_rmse_seg, src == "test"),
    aes(y = val), inherit.aes = FALSE,
    x = factor("test", levels = src_levels),
    color = "black", fill = "yellow", shape = 21, size = 3.4, stroke = 0.7
  ) +
  # Connector: last point -> right-side label column
  geom_segment(
    data = end_annot,
    aes(x = as.numeric(src) + 0.05, xend = x_out - 0.05, y = val, yend = y_label, color = symptom),
    inherit.aes = FALSE, linetype = 3, linewidth = 0.5, alpha = 0.8
  ) +
  # Right-aligned labels (symptom · model | feature)
  geom_text(
    data = end_annot,
    aes(x = x_out, y = y_label, label = label, color = symptom),
    inherit.aes = FALSE, hjust = 0, size = 3.4
  ) +
  scale_x_discrete(labels = lang_labels, expand = expansion(add = c(0.2, 1.8))) +
  labs(x = NULL, y = "RMSE",) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none",
    plot.margin = margin(5, 60, 5, 5)   # add more space on the right for labels
  ) +
  coord_cartesian(clip = "off")

rmse_seg

ggsave("results/Figure_2/D_best_model_RMSE_seg.svg", rmse_seg, width = 12.5, height = 5.5)
ggsave("results/model_comparison/best_model_RMSE_seg.svg", rmse_seg, width = 12.5, height = 5.5)

# Select RMSE + par
df_rmse_par <- df_long %>%
  dplyr::filter(metric == "rmse", method == "par") %>%
  dplyr::mutate(
    src     = factor(src, levels = src_levels),
    symptom = factor(symptom, levels = c("P1","P2","P3","N1","N4","N6","G5","G9"))
  )

# Keep only model, feature, and symptom for annotation
best_tbl_min <- best_tbl %>% dplyr::select(model, feature, symptom)

# Last available point for each line + label text (symptom · model | feature)
end_annot_par <- df_rmse_par %>%
  dplyr::arrange(symptom, src) %>%
  dplyr::group_by(symptom) %>%
  dplyr::filter(!is.na(val)) %>%
  dplyr::slice_tail(n = 1) %>%
  dplyr::ungroup() %>%
  dplyr::left_join(best_tbl_min, by = "symptom") %>%
  dplyr::mutate(label = paste0(symptom, " · ", model, " | ", feature))

# X position for labels (outside the plotting area)
x_out <- length(src_levels) + 0.5

# Arrange labels vertically (from high to low)
end_annot_par <- end_annot_par %>%
  dplyr::arrange(dplyr::desc(val)) %>%
  dplyr::mutate(
    y_label = seq(from = max(val) + 0.05, to = min(val) - 0.05, length.out = dplyr::n())
  )

rmse_par <- ggplot(df_rmse_par, aes(x = src, y = val, group = symptom, color = symptom)) +
  geom_line(linewidth = 1, alpha = 0.9, na.rm = TRUE) +
  geom_point(size = 2.2, alpha = 0.9, na.rm = TRUE) +
  # Highlight Test point
  geom_point(
    data = dplyr::filter(df_rmse_par, src == "test"),
    aes(y = val), inherit.aes = FALSE,
    x = factor("test", levels = src_levels),
    color = "black", fill = "yellow", shape = 21, size = 3.4, stroke = 0.7
  ) +
  # Connector: last point -> right-side label column
  geom_segment(
    data = end_annot_par,
    aes(x = as.numeric(src) + 0.05, xend = x_out - 0.05, y = val, yend = y_label, color = symptom),
    inherit.aes = FALSE, linetype = 3, linewidth = 0.5, alpha = 0.8
  ) +
  # Right-aligned labels (symptom · model | feature)
  geom_text(
    data = end_annot_par,
    aes(x = x_out, y = y_label, label = label, color = symptom),
    inherit.aes = FALSE, hjust = 0, size = 3.4
  ) +
  scale_x_discrete(labels = lang_labels, expand = expansion(add = c(0.2, 1.8))) +
  labs(x = NULL, y = "RMSE",) +
  theme_minimal(base_size = 13) +
  theme(
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(angle = 0, hjust = 0.5, vjust = 1),
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "none",
    plot.margin = margin(5, 60, 5, 5)
  ) +
  coord_cartesian(clip = "off")

rmse_par

ggsave("results/Figure_2/E_best_model_RMSE_par.svg", rmse_par, width = 12.5, height = 5.5)
ggsave("results/model_comparison/best_model_RMSE_par.svg", rmse_par, width = 12.5, height = 5.5)

