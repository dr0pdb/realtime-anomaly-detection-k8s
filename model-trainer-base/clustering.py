import glob
import csv
import os
import pandas as pd

path = "./fastStorage/*.csv"
output_path = "./output/"

sample_size_average = 30 # 30 entries = 9000 ms = 9 second average

data = []

for fname in glob.glob(path):
    with open(fname, 'r') as infh:
        next(infh)
        reader = csv.reader(infh, delimiter=';')

        output = open(output_path + fname[fname[2:].find('/')+3:], "w+")

        timestamp_list = []
        cpu_usage_list = []
        mem_usage_percent_list = []
        disk_read_list = []
        disk_write_list = []
        net_in_list = []
        net_out_list = []

        counter = 0

        for row in reader:

            timestamp = int(row[0])
            cpu_usage = float(row[4])
            mem_capacity = float(row[5])
            mem_usage = float(row[6])
            disk_read = float(row[7])
            disk_write = float(row[8])
            net_in = float(row[9])
            net_out = float(row[10])

            # Append to lists
            timestamp_list.append(timestamp)
            cpu_usage_list.append(cpu_usage)

            if (mem_capacity != 0):
                mem_usage_percent_list.append((mem_usage/mem_capacity)*100.0)
            else:
                mem_usage_percent_list.append(0.0)

            disk_read_list.append(disk_read)
            disk_write_list.append(disk_write)
            net_in_list.append(net_in)
            net_out_list.append(net_out)

            counter+= 1

            if counter >= sample_size_average:
                # Get the averages
                timestamp_avg = sum(timestamp_list)/len(timestamp_list)
                cpu_avg = sum(cpu_usage_list)/len(cpu_usage_list)
                mem_avg = sum(mem_usage_percent_list)/len(mem_usage_percent_list)
                disk_read_avg = sum(disk_read_list)/len(disk_read_list)
                disk_write_avg = sum(disk_write_list)/len(disk_write_list)
                net_in_avg = sum(net_in_list)/len(net_in_list)
                net_out_avg = sum(net_out_list)/len(net_out_list)

                counter = 0
                timestamp_list = []
                cpu_usage_list = []
                mem_usage_percent_list = []
                disk_read_list = []
                disk_write_list = []
                net_in_list = []
                net_out_list = []

                # output.write(str(timestamp_avg) + ';' + str(cpu_avg) + ';' + str(mem_avg) + ';' + str(disk_read_avg) + ';' + str(disk_write_avg) + ';' + str(net_in_avg) + ';' + str(net_out_avg) + '\n')
                data.append([timestamp_avg, cpu_avg, mem_avg, disk_write_avg])


df = pd.DataFrame(data, columns = ['timestamp', 'cpu', 'memory', 'disk']) 