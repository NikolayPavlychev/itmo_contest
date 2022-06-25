#Function OnePreprocessing() convert dataset, contain categorical features and transform them into OHE form.
import pandas as pd

def OhePreprocessing(dataset, target=True, train=True, cat_dummies = None, train_cols_order = None):
    cols=list(dataset.columns)

    if target:
        dataset_ohe_form = dataset[['ISU', 'DISC_ID', 'TYPE_NAME','DEBT']]
    else:
        dataset_ohe_form = dataset[['ISU', 'DISC_ID', 'TYPE_NAME']]

    cols.remove('ISU')
    cols.remove('DISC_ID')
    cols.remove('TYPE_NAME')
    if 'DEBT' in cols:
        cols.remove('DEBT')

    for col in cols:
        if pd.api.types.is_object_dtype(dataset.col):
            df = pd.get_dummies(dataset.col, prefix_sep="__",
                              columns=dataset.col)
        else:
            df = pd.DataFrame(dataset.col)
        dataset_ohe_form = pd.concat((dataset.col,df),axis=1)

    if train:
        cat_dummies = [col for col in dataset_ohe_form
                   if "__" in col
                   and col.split("__")[0] in cols]
        train_cols_order = list(dataset_ohe_form.columns)

    else:
        for col in dataset_ohe_form.columns:
            if ("__" in col) and (col.split("__")[0] in cols) and col not in cat_dummies:
                print("Removing additional feature {}".format(col))
                dataset_ohe_form.drop(col, axis=1, inplace=True)

        for col in cat_dummies:
            if col not in dataset_ohe_form.columns:
                print("Adding missing feature {}".format(col))
                dataset_ohe_form[col] = 0

        dataset_ohe_form = dataset_ohe_form[train_cols_order]

    if train:
        return dataset_ohe_form, cat_dummies, train_cols_order
    else:
        return dataset_ohe_form