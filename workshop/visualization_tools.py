import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns




def create_visualization(data, type, target, figsize=(12, 5), bins=50):
    if type == 'quantitative':
        print('Visualization of quantitative features distribution:')
        print()

        for i in data.drop(target, axis=1):
            if data[i].dtype != 'O' and list(data[i].unique()) != [0, 1]:
                print('-----------------------------------------------------------------------------------------------')
                print(f'Feature: {i}')

                figure, (ax_box, ax_hist) = plt.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": (.15, .85)},
                                                         figsize=figsize)

                sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)

                sns.boxplot(data[i], ax=ax_box).set_title(i, y=1.5, fontsize=14)
                sns.histplot(data[i], bins=bins, kde=True, ax=ax_hist)

                plt.show()
                print()
                print('Feature statistics:')
                display(data[i].describe())
    elif type == 'category':
        print('Visualization of categorical features and target prevalence:')
        print()

        for i in data.drop(target, axis=1):
            if data[i].dtype == 'O' or list(data[i].unique()) == [0, 1]:
                bar_data = data[i].value_counts().reset_index().rename(
                    columns={'index': data[i].name, data[i].name: 'Count'}).sort_values(by=i)
                prev_data = data.pivot_table(index=i, values=target).reset_index().sort_values(by=i)

                print('-----------------------------------------------------------------------------------------------')
                print(f'Feature: {i}')

                figure, (ax_bar, ax_prev) = plt.subplots(1, 2, figsize=figsize)

                sns.barplot(
                    data=bar_data,
                    x=bar_data[data[i].name], y=bar_data['Count'],
                    ax=ax_bar
                ).set_title(i, y=1.02, fontsize=14)
                sns.barplot(
                    data=prev_data,
                    x=prev_data[i], y=prev_data[target],
                    ax=ax_prev
                ).set_title(f'{i}. Target prevalence', y=1.02, fontsize=14)

                plt.show()
                print()
                print('Feature statistics:')
                display(data[i].describe())
    elif type == 'target':
        print('Visualization of target distribution:')
        print()
        print(f'Target: {target}')

        plt.figure(figsize=(7, 5))

        sns.barplot(
            data=data[target].value_counts().reset_index().rename(
                columns={'index': data[target].name, data[target].name: 'Count'}),
            x=data[target], y='Count'
        ).set_title(target, y=1.02, fontsize=14)

        plt.show()
        print()
        print('Target statistics:')
        display(data[target].describe())


def get_corr_map(df, method='pearson', figisze=(15, 12)):
    plt.figure(figsize=figisze)

    sns.heatmap(
        round(df.corr(method=method), 2), vmax=1, vmin=-1, square=True, linewidths=3, annot=True, cmap='coolwarm'
    )

    plt.show()


def create_metrics(model, X, y, only_result):
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, precision_recall_curve, roc_curve

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    results = pd.DataFrame(
        {
            'accuracy': [accuracy_score(y, predictions)],
            'precision': [precision_score(y, predictions)],
            'recall': [recall_score(y, predictions)],
            'f1': [f1_score(y, predictions)],
            'auc': [roc_auc_score(y, probabilities)],
        }
    ).T.reset_index().rename(columns={'index': 'metrics', 0: 'score'})

    print('Metrics values:')

    if only_result:
        return results
    else:
        display(results)

        conf_matrix = pd.DataFrame(confusion_matrix(y, predictions))
        conf_matrix_norm = pd.DataFrame(confusion_matrix(y, predictions, normalize='true') * 100)

        print('Confusion Matrix:')
        display(conf_matrix)
        print()
        # print('Confusion Matrix, %')
        # display(conf_matrix_norm)

        figure, (ax_roc, ax_f1, ax_matrix) = plt.subplots(1, 3, figsize=(21, 5))

        fpr, tpr, thresholds = roc_curve(y, probabilities)
        ax_roc.plot(fpr, tpr, lw=2, label='ROC curve ')
        ax_roc.plot([0, 1], [0, 1])
        ax_roc.set_xlim([-0.05, 1.0])
        ax_roc.set_ylim([0.0, 1.05])
        ax_roc.set_xlabel('False Positive Rate')
        ax_roc.set_ylabel('True Positive Rate')
        ax_roc.set_title('ROC curve')

        precision, recall, thresholds = precision_recall_curve(y, probabilities)

        ax_f1.step(recall, precision, where='post')
        ax_f1.set_xlabel('Recall')
        ax_f1.set_ylabel('Precision')
        ax_f1.set_xlim([-0.05, 1.0])
        ax_f1.set_ylim([0.0, 1.05])
        ax_f1.set_title('Precision-Recall')

        sns.heatmap(
            conf_matrix_norm,
            linewidths=3,
            linecolor='white',
            annot=True,
            cmap='Blues',
            ax=ax_matrix,
        )
        ax_matrix.set_title('Confusion Matrix')

        print('Visualization of metrics:')
        plt.show()


def get_feature_importances(model, X_train):

    if model.feature_importances_.shape != ():
        print('Feature importances:')

        feature_importances = pd.Series(
            model.feature_importances_, index=X_train.columns
        ).reset_index().rename(
            columns={'index': 'feature', 0: 'importance'}
        ).sort_values(
            by='importance', ascending=False
        ).reset_index(drop=True)

        feature_importances['feature'] = feature_importances['feature'].apply(str)

        plt.figure(figsize=(10, 10))
        sns.barplot(y=feature_importances['feature'], x=feature_importances['importance'])

        plt.show()
    else:
        print('Cannot calculate feature importances for loaded models!')


def get_shap(model, X, y, plot_size=(12, 10)):
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, y)

    shap.summary_plot(shap_values, X, plot_size=plot_size)