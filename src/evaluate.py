import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


def evaluate_model(pipe, X_test, y_test, name: str) -> dict:
    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f'\n=== {name} ===')
    print(f'Accuracy : {acc:.4f} | ROC-AUC : {auc:.4f}')
    print(classification_report(y_test, y_pred, target_names=['Good (0)', 'Bad (1)']))
    return {'y_pred': y_pred, 'y_proba': y_proba, 'acc': acc, 'auc': auc}


def select_best(eval_results: dict) -> str:
    return max(eval_results, key=lambda k: eval_results[k]['auc'])


def summary_table(eval_results: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {'Model': name, 'Accuracy': round(res['acc'], 4), 'ROC-AUC': round(res['auc'], 4)}
        for name, res in eval_results.items()
    ]).sort_values('ROC-AUC', ascending=False).reset_index(drop=True)
