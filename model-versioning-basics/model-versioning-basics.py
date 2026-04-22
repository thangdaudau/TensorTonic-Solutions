def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    # Write code here
    best = models[0]
    for model in models[1:]:
        if model["accuracy"] > best["accuracy"] \
            or model["accuracy"] == best["accuracy"] and model["latency"] < best["latency"] \
            or model["accuracy"] == best["accuracy"] and model["latency"] == best["latency"] and model["timestamp"] > best["timestamp"]:
            best = model
    return best["name"]