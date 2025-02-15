import os
import pandas as pd
import openai
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes.models import (
    SearchIndex, SimpleField, SearchableField, SearchFieldDataType,SearchField,
    VectorSearch, HnswAlgorithmConfiguration, HnswParameters, VectorSearchProfile
)
from dotenv import load_dotenv

load_dotenv()

# 環境変数からAzureのAPIキーを取得
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_SEARCH_API_KEY = os.getenv("AZURE_SEARCH_API_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX_NAME = os.getenv("AZURE_SEARCH_INDEX_NAME")

# Azure AI Search クライアント設定
search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX_NAME,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

# インデックス作成クライアント設定
index_client = SearchIndexClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    credential=AzureKeyCredential(AZURE_SEARCH_API_KEY)
)

# インデックス作成
def create_index():
    print("インデックス作成開始")
    index = SearchIndex(
        name=AZURE_SEARCH_INDEX_NAME,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True),
            SearchableField(name="name", type=SearchFieldDataType.String, analyzer="ja.microsoft", searchable=True),
            SearchableField(name="birth_date", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="education", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="research_field", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="research_achievements", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="awards", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="self_intro", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="appeal", type=SearchFieldDataType.String, searchable=True),
            SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                        vector_search_dimensions=3072, vector_search_profile_name="vector_profile")
        ],
        vector_search=VectorSearch(
            algorithms=[
                HnswAlgorithmConfiguration(
                    name="my-hnsw-config",
                    parameters=HnswParameters(
                        m=4,
                        ef_construction=400,
                        ef_search=500,
                        metric="cosine"
                    )
                )
            ],
            profiles=[
                VectorSearchProfile(
                    name="vector_profile",
                    algorithm_configuration_name="my-hnsw-config"
                )
            ]
        )
    )
    
    index_client.create_or_update_index(index)
    print(f"インデックス '{AZURE_SEARCH_INDEX_NAME}' を作成しました！")

# インデックス作成を実行
create_index()

# OpenAI API クライアント設定
openai.api_key = AZURE_OPENAI_API_KEY
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-07-01-preview"

# CSVデータの読み込み
df = pd.read_csv("researchers.csv")

# 埋め込みを取得する関数
def get_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response.data[0].embedding

# 研究者データをAzure AI Searchに登録
documents = []
for idx, row in df.iterrows():
    # 埋め込み取得
    embedding = get_embedding(row["氏名"] + " " + row["研究分野"] + " " + row["研究実績"])
    
    # Azure AI Search 用のドキュメント作成
    document = {
        "id": str(idx + 1),  # 連番をIDとして使用
        "name": row["氏名"],
        "birth_date": row["生年月日"],
        "education": row["学歴"],
        "research_field": row["研究分野"],
        "research_achievements": row["研究実績"],
        "awards": row["表彰実績"],
        "self_intro": row["自己紹介"],
        "appeal": row["アピール"],
        "vector": embedding  # 埋め込みベクトル
    }
    documents.append(document)

# インデックスにデータをアップロード
print("データアップロード開始")
search_client.upload_documents(documents=documents)
print(f"{len(documents)} 件の研究者データをAzure AI Searchに登録しました！")
