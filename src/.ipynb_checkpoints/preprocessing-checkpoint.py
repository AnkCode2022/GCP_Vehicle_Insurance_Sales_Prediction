# import pandas as pd
# import yaml
# import os

# def load_config():
#     with open("config/config.yaml") as f:
#         return yaml.safe_load(f)

# def preprocess(df):
#     df = df.copy()

#     # Encode categorical
#     df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
#     df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

#     df['Vehicle_Age'] = df['Vehicle_Age'].map({
#         '< 1 Year': 0,
#         '1-2 Year': 1,
#         '> 2 Years': 2
#     })

#     return df

# def run():
#     config = load_config()

#     train = pd.read_csv(config['paths']['train_local'])
#     test = pd.read_csv(config['paths']['test_local'])

#     train = preprocess(train)
#     test = preprocess(test)

#     os.makedirs("artifacts/processed", exist_ok=True)

#     train.to_csv(config['paths']['processed_train'], index=False)
#     test.to_csv(config['paths']['processed_test'], index=False)

#     print("Preprocessing done")

# if __name__ == "__main__":
#     run()



import pandas as pd
import yaml
import os

def load_config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)

def preprocess(df):
    df = df.copy()

    # Drop ID column if present
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    # Encode categorical
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})

    df['Vehicle_Age'] = df['Vehicle_Age'].map({
        '< 1 Year': 0,
        '1-2 Year': 1,
        '> 2 Years': 2
    })

    return df

def run():
    config = load_config()

    train = pd.read_csv(config['paths']['train_local'])
    test = pd.read_csv(config['paths']['test_local'])

    train = preprocess(train)
    test = preprocess(test)

    os.makedirs("artifacts/processed", exist_ok=True)

    train.to_csv(config['paths']['processed_train'], index=False)
    test.to_csv(config['paths']['processed_test'], index=False)

    print("✅ Preprocessing done")

if __name__ == "__main__":
    run()