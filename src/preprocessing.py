import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class Preprocessor:
    def __init__(self):
        self.num_features = ['price', 'nb_rooms', 'salon', 'nb_baths', 'surface_area']
        self.cat_feature = 'city_name'
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def load_data(self, path):
        df = pd.read_csv(path)
        print(f"Chargement des données: {df.shape[0]} lignes, {df.shape[1]} colonnes")
        return df

    def remove_outliers_iqr(self, df, features):
        # Méthode pour supprimer les outliers selon IQR sur colonnes numériques
        for col in features:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        return df

    def clean_and_transform(self, df):
        # Nettoyer prix
        df['price'] = df['price'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df['price'] = df['price'].fillna(df['price'].median())
        df = df[df['price'] > 0]
        df['log_price'] = np.log(df['price'].astype(float))

        # Harmoniser villes (comme avant)
        arabic_to_french = {'الدار البيضاء': 'Casablanca', 'الرباط': 'Rabat'}
        df[self.cat_feature] = df[self.cat_feature].replace(arabic_to_french).fillna('Unknown')
        df[self.cat_feature] = self.cat_imputer.fit_transform(df[[self.cat_feature]]).ravel()

        # One-hot encoding ville
        ohe_result = self.ohe.fit_transform(df[[self.cat_feature]])
        ohe_cols = [f"city_{cat}" for cat in self.ohe.categories_[0]]
        ohe_df = pd.DataFrame(ohe_result, columns=ohe_cols)

        # Garder uniquement colonnes principales + one-hot ville
        df = pd.concat([df.reset_index(drop=True), ohe_df], axis=1)
        df = df[self.num_features + ohe_cols + ['log_price']]

        # Nettoyer les outliers sur colonnes numériques
        df = self.remove_outliers_iqr(df, self.num_features)

        # Imputation + standardisation des numériques
        df[self.num_features] = self.num_imputer.fit_transform(df[self.num_features])
        df[self.num_features] = self.scaler.fit_transform(df[self.num_features])

        print(f"Données nettoyées (sans outliers) : {df.shape[0]} lignes, {df.shape[1]} colonnes")
        self.is_fitted = True
        return df

    def select_features(self, df):
        X = df.drop(columns=['log_price'])
        y = df['log_price']
        return X, y

    def transform_for_prediction(self, df_input):
        df = df_input.copy()

        # Harmoniser ville + imputer
        df[self.cat_feature] = df[self.cat_feature].replace({'الدار البيضاء': 'Casablanca', 'الرباط': 'Rabat'}).fillna('Unknown')
        df[self.cat_feature] = self.cat_imputer.transform(df[[self.cat_feature]]).ravel()

        # One-hot encoding
        ohe_result = self.ohe.transform(df[[self.cat_feature]])
        ohe_cols = [f"city_{cat}" for cat in self.ohe.categories_[0]]
        ohe_df = pd.DataFrame(ohe_result, columns=ohe_cols)
        df = pd.concat([df.reset_index(drop=True), ohe_df], axis=1)
        df.drop(columns=[self.cat_feature], inplace=True)

        # Garder uniquement colonnes principales + villes one-hot
        for col in self.num_features:
            if col not in df.columns:
                df[col] = 0

        df = df[self.num_features + ohe_cols]

        # Imputer et scaler
        df[self.num_features] = self.num_imputer.transform(df[self.num_features])
        df[self.num_features] = self.scaler.transform(df[self.num_features])

        return df
