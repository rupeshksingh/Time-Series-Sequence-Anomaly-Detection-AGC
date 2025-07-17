def fill_anomaly_gaps(predictions, gap_threshold=10):
    """
    Fill gaps between anomaly segments if they are smaller than threshold.
    """
    filled_predictions = predictions.copy()
    anomaly_starts, anomaly_ends = [], []
    in_anomaly = False
    for i in range(len(predictions)):
        if predictions[i] == 1 and not in_anomaly:
            anomaly_starts.append(i)
            in_anomaly = True
        elif predictions[i] == 0 and in_anomaly:
            anomaly_ends.append(i - 1)
            in_anomaly = False
    if in_anomaly:
        anomaly_ends.append(len(predictions) - 1)

    for i in range(len(anomaly_ends) - 1):
        gap_start = anomaly_ends[i] + 1
        gap_end = anomaly_starts[i + 1] - 1
        gap_size = gap_end - gap_start + 1
        if gap_size <= gap_threshold:
            filled_predictions[gap_start:gap_end + 1] = 1
    return filled_predictions