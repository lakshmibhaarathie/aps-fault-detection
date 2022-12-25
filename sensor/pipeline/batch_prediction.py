import os

import pandas as pd
import numpy as np
from datetime import datetime
from sensor.model_resolver import ModelResolver
from sensor.exception import SensorException
from sensor import utils


def start_batch_prediction(file_path):
    try:
        prediction_dir = os.path.join("prediction")
        os.makedirs(prediction_dir, exist_ok=True)
        model_resolver = ModelResolver()
        df = pd.read_csv(file_path)
        df.replace({"na": np.NAN}, inplace=True)

        # get model objects path
        transformer_path = model_resolver.get_latest_registry_transformer()
        target_encoder_path = model_resolver.get_latest_registry_target_encoder()
        model_path = model_resolver.get_latest_registry_model()

        # load model objects

        transformer = utils.load_file_object(file_path=transformer_path)
        target_encoder = utils.load_file_object(file_path=target_encoder_path)
        model = utils.load_file_object(file_path=model_path)

        input_feature_names = list(transformer.feature_names_in_)
        input_arr = transformer.transform(df[input_feature_names])

        y_pred = model.predict(input_arr)
        predicted_category = target_encoder.inverse_transform(y_pred)

        df["target"] = y_pred
        df["target_category"] = predicted_category

        pred_file_name = os.path.basename(file_path)
        new_pred_file_name = pred_file_name.replace(".csv", f"{datetime.now().strftime('%d%m%Y__%H%M%S')}.csv")
        pred_file_path = os.path.join(prediction_dir, new_pred_file_name)
        df.to_csv(path_or_buf=pred_file_path, index=False, header=True)
        return pred_file_path
    except Exception as e:
        raise SensorException(e)
