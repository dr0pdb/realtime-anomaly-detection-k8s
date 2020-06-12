from trainer.py import train_model
import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
import subprocess

# Azure Account Keys
connect_str = "DefaultEndpointsProtocol=https;AccountName=btpstorage;AccountKey=SLMZB4+BfyT5V06NHKnJfUI/fgM1Och2u5sC2U0Lgwt7LPwvUKZ16V1OrvHIvcoBh3Lv/MqjPRPEsGAQ+ER1HQ==;EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connect_str)
model_container_name = "models"
dataset_table_name = "datasets"
dataset_table_service = TableService(account_name='btpstorage', account_key='SLMZB4+BfyT5V06NHKnJfUI/fgM1Och2u5sC2U0Lgwt7LPwvUKZ16V1OrvHIvcoBh3Lv/MqjPRPEsGAQ+ER1HQ==')
model_blob_client = blob_service_client.get_blob_client(container=model_container_name, blob="models")


def download_data():
	dataset = dataset_table_service.query_entities(dataset_table_name)
    for data in dataset:
        print(data.timestamp)
        print(data.usage)

    table_service.delete_table(dataset_table_name)


def download_pre_trained_model():
	pass


def build_container_and_upload():
	pass


if __name__ == "__main__":
	print("Downloading dataset...")
    dataset = download_data()

    print("Downloading pre trained model...")
    pre_trained_model = download_pre_trained_model()

    print("Training pre trained model on new data...")
    updated_model = train_model(pre_trained_model, dataset)

    print("Building container and uploading to dockerhub...")
    build_container_and_upload()
