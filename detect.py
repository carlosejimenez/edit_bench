def detect(
    frame: np.ndarray, model: RoboflowInferenceModel, confidence_threshold: float = 0.5
) -> sv.Detections:
    """
    Detect objects in a frame using Inference model, filtering detections by class ID
        and confidence threshold.

    Args:
        frame (np.ndarray): The frame to process, expected to be a NumPy array.
        model (RoboflowInferenceModel): The Inference model used for processing the
            frame.
        confidence_threshold (float, optional): The confidence threshold for filtering
            detections. Default is 0.5.

    Returns:
        sv.Detections: Filtered detections after processing the frame with the Inference
            model.

    Note:
        This function is specifically tailored for an Inference model and assumes class
        ID 0 for filtering.
    """
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    filter_by_class = detections.class_id == 0
    filter_by_confidence = detections.confidence > confidence_threshold
    return detections[filter_by_class & filter_by_confidence]