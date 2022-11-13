import processing

raw_data_path = r"./data/data_raw.csv"

data = processing.read_csv_data(raw_data_path)

# ----- data analysis
data.info()
processing.show_histograms(data)
processing.show_corr_matrix(data)

# ----- pipeline
data = processing.remove_missing_values(data)
data = processing.reduce_dataset(data, 10000)
processing.save_to_csv(data, r"./data/data_reduced.csv")

train_data, test_data = processing.split_train_test(data)
processing.save_to_csv(train_data, r"./data/data_reduced_train.csv")
processing.save_to_csv(test_data, r"./data/data_reduced_test.csv")
