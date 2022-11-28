def get_trained_model(X_train, y_train, X_val, y_val, cat_features, random_state, task_type='CPU', gridsearch=False, save=False):
    from catboost import CatBoostClassifier

    if gridsearch:
        model = CatBoostClassifier(
            random_state=random_state,
            task_type=task_type,
            auto_class_weights='Balanced',
            early_stopping_rounds=100,
            verbose=False,

            boosting_type='Ordered',
            cat_features=cat_features,
        )

        grid = {
            'depth': [4],
            'l2_leaf_reg': [1],
            'learning_rate': [0.01, 0.3],
            'iterations': [300, 3000]
        }

        search_res = model.grid_search(grid,
                                       X=X_train,
                                       y=y_train,
                                       calc_cv_statistics=True,
                                       verbose=False,
                                       plot=True)
        model = CatBoostClassifier(
            random_state=random_state,
            task_type=task_type,
            use_best_model=True,
            auto_class_weights='Balanced',
            early_stopping_rounds=100,

            boosting_type='Ordered',
            cat_features=cat_features,

            depth=search_res['params']['depth'],
            l2_leaf_reg=search_res['params']['l2_leaf_reg'],
            learning_rate=search_res['params']['learning_rate'],
            iterations=search_res['params']['iterations'],

            verbose=False,
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, plot=True)
    else:
        model = CatBoostClassifier(
            random_state=random_state,
            task_type=task_type,
            use_best_model=True,
            auto_class_weights='Balanced',
            early_stopping_rounds=150,

            boosting_type='Ordered',
            cat_features=cat_features,

            depth=6,
            learning_rate=0.003,
            iterations=5000,

            verbose=False
        )

        model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False, plot=True)

    if save:
        import datetime
        save_path = f'.\models\model {int(round(model.score(X_val, y_val), 4) * 10000)} {datetime.datetime.now().strftime("%d-%m-%Y %H-%M")}.json'

        model.save_model(f'{save_path}', format='json')

    print('\n')
    print('The model has been successfully trained!')
    return model


def create_cv(model, X, y, cat_features, folds, random_state):
    from catboost import Pool, cv

    params = {
        'random_state':             random_state,
        # 'task_type':                model.get_params()['task_type'],
        'use_best_model':           model.get_params()['use_best_model'],
        'auto_class_weights':       model.get_params()['auto_class_weights'],
        'early_stopping_rounds':    150,

        'loss_function':            'Logloss',
        'custom_loss':              'AUC',

        'iterations':               model.get_params()['iterations'],
        'depth':                    model.get_params()['depth'],
        'learning_rate':            model.get_params()['learning_rate'],
        'verbose':                  False}
    cv_dataset = Pool(data=X,
                      label=y,
                      cat_features=cat_features)
    scores = cv(cv_dataset,
                params,
                fold_count=folds,
                plot="True")

    return scores