#Function OnePreprocessing() convert dataset, contain categorical features and transform them into OHE form.

import pandas as pd

def OhePreprocessing(dataset):
    cols=list(dataset.columns)
    dataset_ohe_form = dataset[['OPEID6', 'INSTNM']]

    cols.remove('OPEID6')
    cols.remove('INSTNM')

    for col in cols:
        if pd.api.types.is_object_dtype(dataset[col]):
            df = pd.get_dummies(dataset[col], prefix=str(col), prefix_sep="__",
                              columns=dataset[col])
        else:
            df = pd.DataFrame(dataset[col])
        dataset_ohe_form = pd.concat((dataset_ohe_form,df),axis=1)

    return dataset_ohe_form