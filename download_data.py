import logging
from utils import download_and_extract, process_data

def download_and_prepare_data(config):
    images_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz'
    labels_url = 'https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat'

    images_dir = download_and_extract(images_url, config["data"]["images_dir"])
    labels_path = download_and_extract(labels_url, config["data"]["labels_dir"])

    train_df, val_df, test_df = process_data(images_dir, labels_path, config)
    logging.info(
        f"TRAIN dataset size: {len(train_df)}, "
        f"VAL dataset size: {len(val_df)}, "
        f"Test dataset size: {len(test_df)}"
    )
    return images_dir, train_df, val_df, test_df
