import psutil
import time
import datetime
from azure.cosmosdb.table.tableservice import TableService
from azure.cosmosdb.table.models import Entity
from azure.cosmosdb.table.tablebatch import TableBatch

# properties
data = []
sleep_interval = 300 # 5 minutes
max_size = 1000
model_server_ip_address = '' # Kubernetes Service

def upload_data_to_azure(table_service):
	global data

	# Create table if it doesn't exist
	table_service.create_table(dataset_table_name, fail_on_exist=False)

	batch = TableBatch()

	for d in data:
		batch.insert_entity(d)

	table_service.commit_batch(dataset_table_name, batch)
	data = []


# Send cpu usage to the model and check for anomaly
def check_for_anomaly(current_usage):
	print(str(current_usage))


if __name__ == '__main__':
	table_service = TableService(account_name, account_key)

	while True:
		cpu_usage = psutil.cpu_percent()
		ts = time.time()
		st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
		d = {'timestamp': st, 'value': cpu_usage}
		check_for_anomaly(d)
		data.append(d)

		if len(data) == max_size:
			upload_data_to_azure(table_service)

		time.sleep(sleep_interval)