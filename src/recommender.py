import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_data(filepath):
    # Memuat dataset dengan penanganan path dinamis
    if not os.path.exists(filepath):
        # Coba fallback ke path alternatif jika dijalankan dari root atau src
        if os.path.exists("data/laptops_cleaned_v1.csv"):
            filepath = "data/laptops_cleaned_v1.csv"
        elif os.path.exists("../data/laptops_cleaned_v1.csv"):
            filepath = "../data/laptops_cleaned_v1.csv"
        elif os.path.exists("laptops_cleaned_v1.csv"):
            filepath = "laptops_cleaned_v1.csv"
            
    df = pd.read_csv(filepath)
    print(f"Total baris awal: {len(df)}")

    # Kolom fitur komprehensif berdasarkan struktur dataset aktual
    target_columns = [
        'brand_name', 'model', 'price', 'rating', 'processor_brand', 
        'ram_num', 'memory_size', 'memory_type', 'gpu_brand', 
        'gpu_type', 'display_size', 'Touch_Screen'
    ]

    available_cols = [col for col in target_columns if col in df.columns]
    features_df = df[available_cols].copy()

    # Tangani missing values
    features_df = features_df.dropna()

    # Pastikan tipe data numerik bersih
    features_df['price_clean'] = pd.to_numeric(features_df['price'], errors='coerce')
    features_df = features_df.dropna(subset=['price_clean'])

    # Reset index agar sinkron saat .iloc dipanggil setelah filtering
    features_df = features_df.reset_index(drop=True)
    return features_df

def build_content_based_recommendation_matrix(df):
    # Klasifikasi fitur kategorikal dan numerik
    categorical_features = ['brand_name', 'processor_brand', 'memory_type', 'gpu_brand', 'gpu_type', 'Touch_Screen']
    numerical_features = ['price_clean', 'rating', 'ram_num', 'memory_size', 'display_size']

    existing_cat = [col for col in categorical_features if col in df.columns]
    existing_num = [col for col in numerical_features if col in df.columns]

    # ColumnTransformer untuk memproses fitur kategorikal (One-Hot) & numerik (MinMax)
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), existing_cat),
            ('num', MinMaxScaler(), existing_num)
        ]
    )

    feature_matrix = preprocessor.fit_transform(df)
    return feature_matrix, preprocessor

def get_recommendations_by_laptop(laptop_index, df, feature_matrix, top_n=5):
    # Perhitungan Cosine Similarity antar seluruh laptop
    cosine_sim = cosine_similarity(feature_matrix, feature_matrix)

    sim_scores = list(enumerate(cosine_sim[laptop_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]

    laptop_indices = [i[0] for i in sim_scores]
    similarity_scores = [i[1] for i in sim_scores]

    recommended_df = df.iloc[laptop_indices].copy()
    recommended_df['similarity_score'] = similarity_scores

    return recommended_df

if __name__ == "__main__":
    # Tentukan path file secara aman relatif terhadap posisi script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "../data/laptops_cleaned_v1.csv")

    data = load_and_preprocess_data(file_path)
    matrix, transformer = build_content_based_recommendation_matrix(data)
    print(f"Dimensi Matriks Fitur Keseluruhan: {matrix.shape}")

    sample_index = 0
    print(f"\nMencari rekomendasi mirip dengan: {data.iloc[sample_index]['model']}")
    recs = get_recommendations_by_laptop(sample_index, data, matrix, top_n=5)

    print("\nHasil Rekomendasi Laptop:")
    print(recs[['model', 'price_clean', 'ram_num', 'memory_size', 'similarity_score']])