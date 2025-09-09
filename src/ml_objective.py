from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def lr_objective(trial, train_embeddings, train_labels, valid_embeddings, valid_labels):
    # Suggest hyperparameters
    C = trial.suggest_float("C", 1e-5, 10, log=True)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"])
    solver = trial.suggest_categorical("solver", ["liblinear", "saga", "lbfgs"])
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])

    # Convert "none" to None for penalty
    if penalty == "none":
        penalty = None

    # Handle solver-penalty compatibility
    if solver == "liblinear" and penalty in ["elasticnet", None]:
        penalty = "l2"  # liblinear only supports l1, l2
    if solver == "lbfgs" and penalty in ["l1", "elasticnet"]:
        penalty = "l2"  # lbfgs doesn't support l1, elasticnet
    if penalty == "elasticnet":
        solver = "saga"  # only saga supports elasticnet

    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0.1, 0.9)

    # Create model params
    params = {
        "C": C,
        "penalty": penalty,
        "solver": solver,
        "class_weight": class_weight,
        "max_iter": 10000,
        "random_state": 42,
    }

    if l1_ratio is not None:
        params["l1_ratio"] = l1_ratio

    try:
        model = LogisticRegression(**params)
        model.fit(train_embeddings, train_labels)

        # Evaluate
        preds = model.predict(valid_embeddings)
        return accuracy_score(valid_labels, preds)

    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0  # Return poor score for failed trials


def rf_objective(trial, train_embeddings, train_labels, valid_embeddings, valid_labels):
    """Objective function cho Random Forest optimization."""

    # Suggest hyperparameters
    n_estimators = trial.suggest_int("n_estimators", 50, 500, step=50)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
    max_features = trial.suggest_categorical(
        "max_features", ["sqrt", "log2", None, 0.5, 0.7, 0.9]
    )
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    class_weight = trial.suggest_categorical(
        "class_weight", ["balanced", "balanced_subsample", None]
    )

    # Nếu bootstrap=False, không thể dùng oob_score và một số features
    oob_score = False
    if bootstrap:
        oob_score = trial.suggest_categorical("oob_score", [True, False])

    # Suggest criterion
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy", "log_loss"])

    # Advanced parameters
    max_leaf_nodes = trial.suggest_categorical(
        "max_leaf_nodes", [None, 50, 100, 200, 500]
    )
    min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.01)

    # Create model params
    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "class_weight": class_weight,
        "criterion": criterion,
        "max_leaf_nodes": max_leaf_nodes,
        "min_impurity_decrease": min_impurity_decrease,
        "oob_score": oob_score,
        "random_state": 42,
        "n_jobs": -1,  # Use all available cores for faster training
    }

    try:
        model = RandomForestClassifier(**params)
        model.fit(train_embeddings, train_labels)

        # Evaluate
        preds = model.predict(valid_embeddings)
        accuracy = accuracy_score(valid_labels, preds)

        # Optional: Add OOB score as additional metric if available
        if oob_score and bootstrap:
            oob_accuracy = model.oob_score_
            print(
                f"Trial {trial.number}: Validation Acc = {accuracy:.4f}, OOB Acc = {oob_accuracy:.4f}"
            )

        return accuracy

    except Exception as e:
        print(f"RF Trial failed: {e}")
        return 0.0  # Return poor score for failed trials


def cb_objective(trial, train_embeddings, train_labels, valid_embeddings, valid_labels):
    """Objective function cho CatBoost optimization."""

    # Suggest hyperparameters
    iterations = trial.suggest_int("iterations", 100, 1000, step=50)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    depth = trial.suggest_int("depth", 3, 10)
    l2_leaf_reg = trial.suggest_float("l2_leaf_reg", 1, 10)
    border_count = trial.suggest_int("border_count", 32, 255)
    bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 1.0)
    random_strength = trial.suggest_float("random_strength", 0.0, 10.0)

    # Advanced parameters
    subsample = trial.suggest_float("subsample", 0.5, 1.0)
    colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.5, 1.0)
    min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 20)

    params = {
        "iterations": iterations,
        "learning_rate": learning_rate,
        "depth": depth,
        "l2_leaf_reg": l2_leaf_reg,
        "border_count": border_count,
        "bagging_temperature": bagging_temperature,
        "random_strength": random_strength,
        "subsample": subsample,
        "colsample_bylevel": colsample_bylevel,
        "min_data_in_leaf": min_data_in_leaf,
        "random_state": 42,
        "verbose": 0,
        "thread_count": -1,
    }

    try:
        model = CatBoostClassifier(**params)
        model.fit(train_embeddings, train_labels)

        # Evaluate
        preds = model.predict(valid_embeddings)
        return accuracy_score(valid_labels, preds)

    except Exception as e:
        print(f"CatBoost Trial failed: {e}")
        return 0.0
