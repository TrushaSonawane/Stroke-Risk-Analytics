# Load required libraries
pacman::p_load(tidyverse, caret, xgboost, pROC, ROCR, data.table, Matrix)

# Step 1: Load Data
df <- fread("healthcare-dataset-stroke-data.csv")

# Step 2: Data Cleaning
df$bmi <- ifelse(df$bmi == 0, NA, df$bmi)
df$bmi[is.na(df$bmi)] <- median(df$bmi, na.rm = TRUE)

df <- df %>%
  filter(!is.na(gender) & !is.na(smoking_status)) %>%  # Optional: remove rows with unknown
  mutate(
    gender = factor(gender),
    ever_married = factor(ever_married),
    work_type = factor(work_type),
    Residence_type = factor(Residence_type),
    smoking_status = factor(smoking_status),
    stroke = factor(stroke, levels = c(0, 1))
  )

# Step 3: One-Hot Encoding
dummies <- dummyVars(stroke ~ ., data = df)
df_encoded <- data.frame(predict(dummies, newdata = df))
df_encoded$stroke <- df$stroke

# Step 4: Train-Test Split
set.seed(123)
train_idx <- createDataPartition(df_encoded$stroke, p = 0.8, list = FALSE)
train <- df_encoded[train_idx, ]
test <- df_encoded[-train_idx, ]

# Step 5: Prepare DMatrix for XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(train %>% select(-stroke)),
                            label = as.numeric(as.character(train$stroke)))
test_matrix <- xgb.DMatrix(data = as.matrix(test %>% select(-stroke)),
                           label = as.numeric(as.character(test$stroke)))

# Step 6: Cross-Validated XGBoost
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  booster = "gbtree",
  eta = 0.1,
  max_depth = 4
)

set.seed(123)
xgb_cv <- xgb.cv(
  params = params,
  data = train_matrix,
  nrounds = 100,
  nfold = 5,
  verbose = 0,
  early_stopping_rounds = 10,
  print_every_n = 10
)

best_nrounds <- xgb_cv$best_iteration

# Final Model Training
xgb_final <- xgboost(
  params = params,
  data = train_matrix,
  nrounds = best_nrounds,
  verbose = 0
)

# Step 7: Predictions and Evaluation
pred_probs <- predict(xgb_final, test_matrix)
preds <- ifelse(pred_probs > 0.5, 1, 0)

confusion <- confusionMatrix(factor(preds), factor(as.numeric(as.character(test$stroke))))
print(confusion)

# Step 8: AUC Curve
roc_obj <- roc(as.numeric(as.character(test$stroke)), pred_probs)
plot(roc_obj, main = "ROC Curve for XGBoost Stroke Prediction", col = "darkgreen")
cat("AUC:", auc(roc_obj), "\n")

# Step 9: Feature Importance
importance <- xgb.importance(model = xgb_final)
print(importance)
xgb.plot.importance(importance_matrix = importance, top_n = 10, main = "Top 10 Features")

# Step 10: Save Cleaned Data
write.csv(df_encoded, "final_cleaned_stroke_dataset.csv", row.names = FALSE)
